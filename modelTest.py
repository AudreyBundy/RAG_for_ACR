from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# 清空 CUDA 缓存
torch.cuda.empty_cache()

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径
model_path = r'/autodl-fs/data/llama-7b'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# 加载模型并分配到 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 使用半精度
    device_map="auto"          # 自动分配设备
).to(device)

# 创建文本生成 pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 提示文本
prompt = "Explain quick sort"

# 生成输出
output = generator(
    prompt,
    max_length=100,     # 缩短生成长度
    do_sample=True,
    temperature=0.7
)

print(output)