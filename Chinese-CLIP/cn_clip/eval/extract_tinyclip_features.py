# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation. (with single-GPU)
'''
import base64
import os
import argparse
import logging
from io import BytesIO
from pathlib import Path
import json
from PIL import Image

import lmdb
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--extract-image-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract image features."
    )
    parser.add_argument(
        '--extract-text-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract text features."
    )
    parser.add_argument(
        '--image-data', 
        type=str, 
        default="../Multimodal_Retrieval/lmdb/test/imgs", 
        help="If --extract-image-feats is True, specify the path of the LMDB directory storing input image base64 strings."
    )
    parser.add_argument(
        '--text-data', 
        type=str, 
        default="../Multimodal_Retrieval/test_texts.jsonl", 
        help="If --extract-text-feats is True, specify the path of input text Jsonl file."
    )
    parser.add_argument(
        '--image-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output image features."
    )    
    parser.add_argument(
        '--text-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output text features."
    )
    parser.add_argument(
        "--img-batch-size", type=int, default=64, help="Image batch size."
    )
    parser.add_argument(
        "--text-batch-size", type=int, default=64, help="Text batch size."
    )
    parser.add_argument(
        "--context-length", type=int, default=52, help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model-path",
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )    
    args = parser.parse_args()

    return args    


if __name__ == "__main__":
    args = parse_args()

    assert args.extract_image_feats or args.extract_text_feats, "--extract-image-feats and --extract-text-feats cannot both be False!"

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")
    
    args.gpu = 0
    torch.cuda.set_device(args.gpu)

    # Initialize the model.
    # vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    # print('Loading vision model config from', vision_model_config_file)
    # assert os.path.exists(vision_model_config_file)
    #
    # text_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    # print('Loading text model config from', text_model_config_file)
    # assert os.path.exists(text_model_config_file)
    #
    # with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
    #     model_info = json.load(fv)
    #     if isinstance(model_info['vision_layers'], str):
    #         model_info['vision_layers'] = eval(model_info['vision_layers'])
    #     for k, v in json.load(ft).items():
    #         model_info[k] = v

    # model = CLIP(**model_info)
    # convert_weights(model)

    from transformers import CLIPProcessor, CLIPModel

    # 加载 TinyCLIP 模型
    model = CLIPModel.from_pretrained(args.model_path).to(f"cuda:{args.gpu}")
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.model_path)



    class EvalTxtDataset(Dataset):
        def __init__(self, jsonl_filename):
            self.texts = []
            with open(jsonl_filename, "r", encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line.strip())
                    self.texts.append((obj['text_id'], obj['text']))

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]  # 返回 (id, raw_text)


    class EvalImgDataset(Dataset):
        def __init__(self, lmdb_imgs):
            self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False)
            with self.env_imgs.begin() as txn:
                cursor = txn.cursor()
                self.keys = [k for k, _ in cursor if k != b'num_images']
            self.number_images = len(self.keys)

        def __len__(self):
            return self.number_images

        def __getitem__(self, idx):
            with self.env_imgs.begin() as txn:
                image_b64 = txn.get(self.keys[idx])
                image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))).convert("RGB")
            return int(self.keys[idx].decode("utf8")), image

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(args.gpu)
    if args.precision == "fp16":
        convert_weights(model)

    # ====== 2. 构造 DataLoader ======
    txt_dataset = EvalTxtDataset(args.text_data)
    img_dataset = EvalImgDataset(args.image_data)


    def collate_fn_img(batch):
        ids, images = zip(*batch)  # list of ids, list of PIL.Image
        return list(ids), list(images)

    txt_loader = DataLoader(txt_dataset, batch_size=args.text_batch_size, sampler=SequentialSampler(txt_dataset), num_workers=8)
    img_loader = DataLoader(img_dataset, batch_size=args.img_batch_size, sampler=SequentialSampler(img_dataset), num_workers=8,
                            collate_fn=collate_fn_img)

    # Get data.
    if args.extract_image_feats:
        print("Preparing image inference dataset.")
        print(txt_dataset[0])
    if args.extract_text_feats:
        print("Preparing text inference dataset.")
        print(img_dataset[0])
    
    # Resume from a checkpoint.
    # print("Begin to load model checkpoint from {}.".format(args.resume))
    # assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    # # Map model to be loaded to specified single gpu.
    # loc = "cuda:{}".format(args.gpu)
    # checkpoint = torch.load(args.resume, map_location='cpu')
    # start_epoch = checkpoint["epoch"]
    # sd = checkpoint["state_dict"]
    # if next(iter(sd.items()))[0].startswith('module'):
    #     sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    # model.load_state_dict(sd)
    # print(
    #     f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)"
    # )

    # Make inference for texts
    if args.extract_text_feats:
        print('Make inference for texts...')
        if args.text_feat_output_path is None:
            args.text_feat_output_path = "{}.txt_feat.jsonl".format(args.text_data[:-6])
        write_cnt = 0
        with open(args.text_feat_output_path, "w") as fout:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(txt_loader):
                    text_ids, texts = batch
                    text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True,max_length=args.context_length)
                    for k,v in text_inputs.items():
                        text_inputs[k] = v.to(f"cuda:{args.gpu}")
                    text_features = model.get_text_features(**text_inputs)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"text_id": text_id, "feature": text_feature})))
                        write_cnt += 1
        print('{} text features are stored in {}'.format(write_cnt, args.text_feat_output_path))

    # Make inference for images
    if args.extract_image_feats:
        print('Make inference for images...')
        if args.image_feat_output_path is None:
            # by default, we store the image features under the same directory with the text features
            args.image_feat_output_path = "{}.img_feat.jsonl".format(args.text_data.replace("_texts.jsonl", "_imgs"))
        write_cnt = 0
        with open(args.image_feat_output_path, "w") as fout:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(img_loader):
                    image_ids, images = batch
                    image_inputs = processor(images=list(images), return_tensors="pt").to(f"cuda:{args.gpu}")
                    image_features = model.get_image_features(**image_inputs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    for image_id, image_feature in zip(image_ids, image_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                        write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, args.image_feat_output_path))

    print("Done!")