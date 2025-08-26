#
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

split=valid # 指定计算valid或test集特征
#dataset_name=Flickr30k-CN
dataset_name=MUGE
#resume=../clip_cn_vit-b-16.pt
resume=../clip_cn_rn50.pt

python3 -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="./datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="./datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume=${resume} \
    --vision-model=RN50 \
    --text-model=RBT3-chinese

python3 -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="./datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="./datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="./datasets/${dataset_name}/${split}_predictions.jsonl"

python3 cn_clip/eval/evaluation.py \
    ./datasets/${dataset_name}/${split}_texts.jsonl \
    ./datasets/${dataset_name}/${split}_predictions.jsonl \
    output.json
cat output.json
