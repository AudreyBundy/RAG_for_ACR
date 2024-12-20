import faiss
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 加载 CodeBERT 模型和分词器
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

# 获取代码片段的向量表示
def get_code_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # 提取 [CLS] 向量

# 示例数据 - 多个数据项
data_list = [
    {
        "defectType": "OVERLOAD_METHOD_DELETED_ARGS",
        "defectCode": 'String.format("DEF_SYSTEM_FONT_SIZE: %.2f",DEF_SYSTEM_FONT_SIZE,dpi)',
        "fixCode": 'String.format("DEF_SYSTEM_FONT_SIZE: %.2f",DEF_SYSTEM_FONT_SIZE)'
    },
    {
        "defectType": "NULL_POINTER_DEREFERENCE",
        "defectCode": 'obj.method(param1, param2)',
        "fixCode": 'if (obj != null) { obj.method(param1, param2); }'
    },
    # 其他数据项
]

# 创建 Faiss 索引
index = faiss.IndexFlatL2(768)  # 使用 L2 距离度量

# 存储元数据
metadata = {}

# 将多个数据项处理并添加到 Faiss 索引和元数据中
for idx, data in enumerate(data_list):
    defect_code_vector = get_code_embedding(data["defectCode"])
    fix_code_vector = get_code_embedding(data["fixCode"])
    defect_type_vector = get_code_embedding(data["defectType"])

    # 将向量转换为 NumPy 数组格式，Faiss 只支持 NumPy 数组
    def_vector = np.array([defect_code_vector], dtype=np.float32)
    fix_vector = np.array([fix_code_vector], dtype=np.float32)
    defect_type_vector = np.array([defect_type_vector], dtype=np.float32)

    # 将向量添加到索引中
    index.add(def_vector)
    index.add(fix_vector)
    index.add(defect_type_vector)

    # 存储元数据
    metadata[idx * 3] = {"defectType": data["defectType"], "defectCode": data["defectCode"], "fixCode": data["fixCode"]}  # defectCode
    metadata[idx * 3 + 1] = {"defectType": data["defectType"], "defectCode": data["defectCode"], "fixCode": data["fixCode"]}  # fixCode
    metadata[idx * 3 + 2] = {"defectType": data["defectType"], "defectCode": data["defectCode"], "fixCode": data["fixCode"]}  # defectType

# 存储索引
faiss.write_index(index, "code_defect_index.index")

# 存储元数据
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# 检索相似的向量
query_code = 'String.format("DEF_SYSTEM_FONT_SIZE: %.2f", DEF_SYSTEM_FONT_SIZE)'
query_vector = get_code_embedding(query_code)
query_vector = np.array([query_vector], dtype=np.float32)

# 进行检索
D, I = index.search(query_vector, k=1)
print(f"Most similar vector index: {I[0][0]} (distance: {D[0][0]})")

# 获取元数据
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# 输出检索到的缺陷类型、缺陷代码和修复代码
similar_data = metadata[I[0][0]]
print(f"Defect Type: {similar_data['defectType']}")
print(f"Defect Code: {similar_data['defectCode']}")
print(f"Fix Code: {similar_data['fixCode']}")
