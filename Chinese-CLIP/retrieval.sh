split=test # 指定计算valid或test集特征
dataset_name=MUGE
python3 -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="./datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="./datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="./datasets/${dataset_name}/${split}_predictions.jsonl"
