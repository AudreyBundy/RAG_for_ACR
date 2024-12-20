import faiss
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle
import json
import os

# 设置环境变量以避免库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
modele_path = r'D:\GRADUATION\Nju\llm\codebert'

# 加载 CodeBERT 模型和分词器
tokenizer = RobertaTokenizer.from_pretrained(modele_path)
model = RobertaModel.from_pretrained(modele_path)

# 获取代码片段的向量表示
def get_code_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # 提取 [CLS] 向量

# 读取 JSON 数据文件
file_path = r'D:\GRADUATION\Nju\Dataset\MyDataSet\codereviewer_v1\many_withdiff.json'

# 创建 Faiss 索引
index = faiss.IndexFlatL2(768)  # 使用 L2 距离度量，CodeBERT 输出为 768 维向量

# 存储元数据（例如 bugType 和代码片段）
metadata = {}

# 读取 JSON 数据并处理
# 读取 JSON 数据并处理
with open(file_path, 'r', encoding='utf-8') as f:
    data_list = json.load(f)
    for_time = 0
    # 将每条数据添加到 Faiss 索引和元数据中
    for idx, data in enumerate(data_list):
        old = data["old_code"]
        new = data["sourceAfterFix"]
        bug_type = data["bugType"]
        error_code = data["sourceBeforeFix"]  # 提取 sourceBeforeFix

        # 获取 'old'、'new'、'bugType' 和 'error_code' 的向量表示
        old_vector = get_code_embedding(old)
        new_vector = get_code_embedding(new)
        bug_type_vector = get_code_embedding(bug_type)
        error_code_vector = get_code_embedding(error_code)  # 获取 error_code 的向量表示

        # 将向量转换为 NumPy 数组格式（Faiss 只支持 NumPy 数组）
        old_vector = np.array([old_vector], dtype=np.float32)
        new_vector = np.array([new_vector], dtype=np.float32)
        bug_type_vector = np.array([bug_type_vector], dtype=np.float32)
        error_code_vector = np.array([error_code_vector], dtype=np.float32)  # 转换 error_code 向量

        # 将向量添加到 Faiss 索引中
        index.add(old_vector)
        index.add(new_vector)
        index.add(bug_type_vector)
        index.add(error_code_vector)  # 添加 error_code 向量

        # 存储元数据
        metadata[idx * 4] = {"bugType": bug_type, "old": old, "new": new, "error_code": error_code}  # old
        metadata[idx * 4 + 1] = {"bugType": bug_type, "old": old, "new": new, "error_code": error_code}  # new
        metadata[idx * 4 + 2] = {"bugType": bug_type, "old": old, "new": new, "error_code": error_code}  # bugType
        metadata[idx * 4 + 3] = {"bugType": bug_type, "old": old, "new": new, "error_code": error_code}  # error_code

# 存储 Faiss 索引到文件
faiss.write_index(index, "code_defect_index.index")

# 存储元数据
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# 打印向量数据库中存入的总数据条数
print(f"Total vectors stored in Faiss index: {index.ntotal}")

# # # 检索相似的向量
# query_code = 'submittedNode.get("values") != null'  # 示例查询文本
# query_vector = get_code_embedding(query_code)
# query_vector = np.array([query_vector], dtype=np.float32)
#
# # 进行检索，找到最相似的向量
# k = 1  # 返回最相似的 1 个结果
# distances, indices = index.search(query_vector, k)
#
# # 输出检索结果
# print(f"Most similar vector index: {indices[0][0]} (distance: {distances[0][0]})")
#
# # 获取并输出元数据
# with open("metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)
#
# # 根据索引获取最相似的条目
# similar_data = metadata[indices[0][0]]
# print(f"Defect Type: {similar_data['bugType']}")
# print(f"Old Code (before fix): {similar_data['old']}")
# print(f"New Code (after fix): {similar_data['new']}")




# import faiss
# import pickle
# import numpy as np
#
# # 读取 Faiss 索引和元数据
# index = faiss.read_index("code_defect_index.index")
#
# # 读取元数据
# with open("metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)
#
# # 获取 Faiss 索引中的所有向量
# total_vectors = index.ntotal  # 获取存储的向量数量
#
# # 从 Faiss 索引中检索所有向量
# vectors = index.reconstruct_n(0, total_vectors)
#
# # 打印所有的向量及其元数据
# for i in range(0, total_vectors, 4):  # 每四个向量代表一条数据：old, new, bugType, error_code
#     # 获取当前条目的向量
#     old_vector = vectors[i]
#     new_vector = vectors[i + 1]
#     bug_type_vector = vectors[i + 2]
#     error_code_vector = vectors[i + 3]
#
#     # 获取对应的元数据
#     old_data = metadata[i]
#     new_data = metadata[i + 1]
#     bug_type_data = metadata[i + 2]
#     error_code_data = metadata[i + 3]
#
#     # 打印信息
#     print(f"Data {i//4 + 1}:")
#     print(f"  Defect Type: {bug_type_data['bugType']}")
#     print(f"  Old Code (before fix): {old_data['old']}")
#     print(f"  New Code (after fix): {new_data['new']}")
#     print(f"  Error Code: {error_code_data['error_code']}")
#     print(f"  Old Vector: {old_vector[:5]}...")  # 仅打印向量的前5个元素以避免太长
#     print(f"  New Vector: {new_vector[:5]}...")
#     print(f"  Bug Type Vector: {bug_type_vector[:5]}...")
#     print(f"  Error Code Vector: {error_code_vector[:5]}...")
#     print("-" * 50)