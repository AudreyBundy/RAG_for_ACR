import json

from transformers import pipeline
import faiss
import numpy as np
import pickle
from transformers import RobertaTokenizer, RobertaModel
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 编码器模型
modele_path = r'D:\GRADUATION\Nju\llm\codebert'
# 加载 CodeBERT 模型和分词器
tokenizer = RobertaTokenizer.from_pretrained(modele_path)
model = RobertaModel.from_pretrained(modele_path)

# 生成器模型
gen_model_path = r'D:\GRADUATION\Nju\llm\vicuna-7b'
generator = pipeline("text-generation", model=gen_model_path)
# 获取代码片段的向量表示
def get_code_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # 提取 [CLS] 向量
# 拼接检索结果
def get_result():
    input_data_path = r'D:\GRADUATION\Nju\Dataset\ManySStuBs4J\testData'
    with open(input_data_path, 'r') as f:
        data = json.load(f)
    index = faiss.read_index("code_defect_index.index")
    for record in data:
        new_code = record.get('new', '')
        query_vector = get_code_embedding(new_code)
        query_vector = np.array([query_vector], dtype=np.float32)
        # 归一化查询向量
        faiss.normalize_L2(query_vector)
        # 进行检索，返回最相似的k个结果
        D, I = index.search(query_vector, k=2)  # 返回最相似的两个结果
        # 载入元数据
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        result_record = []
        for idx in I[0]:
            result_record.append(metadata[idx])
        record['similarFix'] = result_record
        print(record)
        output_template = {
            "bugType": "",
            "defect_code": "",
            "fix": ""
        }
        prompt =f"""
        The following is a code commit, where new is the submitted code, and I want to detect the defects of this committed code. 
        similarFix is a code fix example similar to the submitted code, where the bugType refers to the defect type of the corresponding code fix, 
        and new refers to the corresponding fixed code. You can refer to similarFix for the answer. {record}
        The final output is output in the following format:{output_template}.The bugType is the type of defect you identify, defect_code is the defect code you identified, and fix is the fix you give
        """
        # 将生成结果存入文件
        output = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
        output_file_path = 'output.json'
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            with open(output_file_path, 'a', encoding='utf-8') as f:
                f.write(',\n' + json.dumps(output))
        else:
            with open(output_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output))

if __name__ == '__main__':
    get_result()




# code_query = """
# """
#
# # 载入已存储的索引和元数据
# index = faiss.read_index("code_defect_index.index")
#
# # 载入元数据
# with open("metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)
#
# # 检索相似的向量
# query_vector = get_code_embedding(code_query)
# query_vector = np.array([query_vector], dtype=np.float32)
#
# # 归一化查询向量
# faiss.normalize_L2(query_vector)
#
# # 进行检索，返回最相似的k个结果
# D, I = index.search(query_vector, k=2)  # 返回最相似的两个结果
#
# # 构建 JSON 输出结果
# output = {
#     "queryCode": code_query.strip(),
#     "similarFix": []
# }
#
# # 添加检索到的缺陷类型和修复代码
# for idx in I[0]:
#     similar_data = metadata[idx]
#     result = {
#         "defectType": similar_data['defectType'],
#         "fixCode": similar_data['fixCode']
#     }
#     output["similarFix"].append(result)

# # 将结果转换为 JSON 格式并打印
# output_json = json.dumps(output, indent=4)
# print(output_json)
#
#
#
# # 本地模型路径
# model_path = r'D:\GRADUATION\Nju\llm\codereviewer'
#
# # 加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# # 编码输入
# inputs = tokenizer(output_json, return_tensors="pt")
#
# # 使用模型生成代码评审评论
# outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
# review_comments = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
# print("Code Review Comments:", review_comments)


