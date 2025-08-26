import torch
import torch.nn as nn

def cal_params(model):
    for name, value in model.named_parameters():
        print(name)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

def count_parameters(model, exclude_key="text_model"):
    total = 0
    for name, param in model.named_parameters():
        if exclude_key not in name:  # 过滤掉包含 "text_model" 的参数
            total += param.numel()
    return total

if __name__ == '__main__':
    from multilingual_clip import pt_multilingual_clip
    import transformers
    import torch
    # from sentence_transformers import SentenceTransformer
    # model_path = "/data/gongoubo/checkpoints/sentence-transformers/clip-ViT-B-32-multilingual-v1"
    # img_model = SentenceTransformer('clip-ViT-B-32')
    # # model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_path)
    # model = SentenceTransformer(model_path)
    # cal_params(img_model)
    # print(count_parameters(img_model))

    model = torch.load("/data/gongoubo/checkpoints/wkcn/clip_cn_rn50.pt")
    total = 0
    for k,v in model["state_dict"].items():
        total += v.numel()
    print(total)