export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

split=test # 指定计算valid或test集特征
dataset_name=MUGE
resume=../clip_cn_vit-b-16.pt
#resume=../clip_cn_rn50.pt

python3 -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="./datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="./datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume=${resume} \
    --vision-model=ViT-B-16 \
    --text-model=RoBERTa-wwm-ext-base-chinese
