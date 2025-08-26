#
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

#TinyCLIP-ViT-39M-16-Text-19M-YFCC15M/
#TinyCLIP-ViT-40M-32-Text-19M-LAION400M/
#TinyCLIP-ViT-61M-32-Text-29M-LAION400M/
#TinyCLIP-ViT-8M-16-Text-3M-YFCC15M/

split=valid # 指定计算valid或test集特征
dataset_name=MUGE
#split=test # 指定计算valid或test集特征
#dataset_name=Flickr30k-CN
#model_path=../TinyCLIP-ViT-39M-16-Text-19M-YFCC15M
model_path=/data/gongoubo/checkpoints/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese

python3 -u cn_clip/eval/extract_taiyi_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="./datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="./datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --context-length 77 \
    --text-batch-size=32 \
    --model-path=${model_path} \

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
