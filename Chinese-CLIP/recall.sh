split=test # 指定计算valid或test集特征
dataset_name=MUGE
python3 cn_clip/eval/evaluation.py \
    ./datasets/${dataset_name}/${split}_texts.jsonl \
    ./datasets/${dataset_name}/${split}_predictions.jsonl \
    output.json
cat output.json
