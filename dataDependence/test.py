import random
import json
import faiss
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle
import os

# 设置环境变量以避免库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print("start")

# 加载模型和分词器
# load_dir = r'/autodl-fs/data/testCode/retriver/RAG_FOR_ACR/save'  # 指定保存模型的路径
load_dir = r'/autodl-fs/data/codebert'
tokenizer = RobertaTokenizer.from_pretrained(load_dir)
model = RobertaModel.from_pretrained(load_dir)
print(load_dir)
# 获取代码片段的向量表示
def get_code_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # 提取 [CLS] 向量

# 读取 JSON 文件中的数据
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 随机抽取指定数量的数据
def sample_data(data, num_samples):
    # 随机获取
    # return random.sample(data, num_samples)
    # 顺序获取
    return data[:num_samples]

# 获取未被选中的数据
def get_remaining_data(data, sampled_data):
    sampled_set = set(tuple(item.items()) for item in sampled_data)  # 转换为元组并放入集合
    return [item for item in data if tuple(item.items()) not in sampled_set]  # 筛选出未选中的数据


# 创建 Faiss 索引并插入数据
# 创建 Faiss 索引并插入 defect_code 的数据
def create_defect_code_index(data1, data2):
    # 初始化专用于 defect_code 的索引
    index = faiss.IndexFlatIP(768)  # 使用点积（Inner Product）索引
    metadata = {}

    for idx, data in enumerate(data1 + data2):
        defect_code = data["sourceBeforeFix"]
        print(defect_code)
        # 获取 defect_code 的向量表示
        defect_code_vector = get_code_embedding(defect_code)
        defect_code_vector = np.array([defect_code_vector], dtype=np.float32)
        faiss.normalize_L2(defect_code_vector)  # 归一化向量

        # 将向量添加到索引中
        index.add(defect_code_vector)

        # 存储元数据（记录 defect_code 和其他信息）
        metadata[idx] = {
            "defect_code": defect_code,
            "bugType": data["bugType"],
            "old_code": data["old_code"],
            "fix_code": data["sourceAfterFix"]
        }

    return index, metadata

def search_defect_code(query_code, index_path, metadata_path, k=5):
    # 加载 Faiss 索引
    index = faiss.read_index(index_path)

    # 加载元数据
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # 获取查询向量
    query_vector = get_code_embedding(query_code)
    query_vector = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_vector)  # 归一化查询向量

    # 检索最相似的结果
    D, I = index.search(query_vector, k)

    # 解析检索结果
    results = []
    for idx in I[0]:
        if idx in metadata:
            results.append(metadata[idx])

    return results
# 主流程
def main():
    # # 文件路径
    # file_path1 = '/autodl-fs/data/dataSet/many_withdiff.json'  # 文件1路径
    # file_path2 = '/autodl-fs/data/dataSet/many_withdiff_largesstubs.json'  # 文件2路径
    #
    # # 读取数据
    # data1 = load_json_data(file_path1)
    # data2 = load_json_data(file_path2)
    #
    # # 随机抽取 5,000 条数据和 30,000 条数据
    # sampled_data1 = sample_data(data1, 100)
    # sampled_data2 = sample_data(data2, 0)
    #
    # # 获取未被选中的数据
    # # remaining_data1 = get_remaining_data(data1, sampled_data1)
    # # remaining_data2 = get_remaining_data(data2, sampled_data2)
    #
    # # 将未选中的数据保存到新的 JSON 文件
    # # with open("remaining_data1.json", "w", encoding="utf-8") as f:
    # #     json.dump(remaining_data1, f, ensure_ascii=False, indent=4)
    #
    # # with open("remaining_data2.json", "w", encoding="utf-8") as f:
    # #     json.dump(remaining_data2, f, ensure_ascii=False, indent=4)
    #
    # # 创建 Faiss 索引并插入抽取的数据
    # index, metadata = create_defect_code_index(sampled_data1, sampled_data2)
    #
    # # 存储 Faiss 索引到文件
    # faiss.write_index(index, "defect_code_index_test.index")
    #
    # # 存储元数据
    # with open("metadata_defect_code_test.pkl", "wb") as f:
    #     pickle.dump(metadata, f)
    #
    # # 打印向量数据库中存入的总数据条数
    # print(f"Total vectors stored in Faiss index: {index.ntotal}")
    q_code = 'map.put(entry.getKey(),valueToString(joiner,entry.getValue()))'
    print(q_code)
    result_re = search_defect_code(q_code, "defect_code_index_test.index", "metadata_defect_code_test.pkl")
    print(result_re)

if __name__ == '__main__':
    main()





