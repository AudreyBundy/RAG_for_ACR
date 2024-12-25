import faiss
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import pickle
import json
import os
from retriver.model import SimpleModel

# 设置环境变量以避免库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
modele_path = r'/autodl-fs/data/codebert'
train_weight_path = r'/autodl-fs/data/retriver/RAG_FOR_ACR/save/contrastive_model.pt'

# 加载 CodeBERT 模型和分词器

# 模型定义
class Args:
    pretrained_dir = "/root/autodl-fs/codebert"          # 预训练模型目录
    vocab_size = 50265                       # 词汇表大小
    num_vec = 1                              # 对比学习中每个样本的向量数
    moco_T = 0.07                            # 温度参数
    learning_rate = 1e-5                     # 学习率
    batch_size = 32                          # 批次大小
    num_epochs = 5                           # 训练轮数
    train_data_path = "contrastive_train_data_v1.json"      # 训练数据路径
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    from_pretrained = True  # 是否从预训练模型加载

tokenizer = RobertaTokenizer.from_pretrained(modele_path)
config = RobertaConfig.from_pretrained(modele_path)
args = Args()
model = SimpleModel(config, args).cuda()
model.load_state_dict(torch.load(train_weight_path))
input = "Status.constructStatuses(get(getBaseURL() + \"favorites/\" + id+ \".json\",new PostParameter[0],true))"
attn_mask1 = torch.tensor(input.clone().detach() != tokenizer.pad_token_id, dtype=torch.uint8, device=args.device)
print(model(input, attn_mask1=attn_mask1))[0]