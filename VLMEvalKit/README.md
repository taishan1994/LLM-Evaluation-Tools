# 安装环境

`pip install -e .`

# 评测

1. 修改vlmeval/inference.py里面的url为vllm部署或者sglang部署后的url。
2. 选择不同的数据集测试即可：`python run.py --data MMBench_DEV_EN --model test-qwen --verbose`
