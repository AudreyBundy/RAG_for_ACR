import json
import faiss
import numpy as np
import pickle
from transformers import RobertaTokenizer, RobertaModel, pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import time

import os

TOP_K = 2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
error_output_file_path = 'error_output.json'
output_file_path = 'output.json'

# 编码器模型
em_model_path = r'/autodl-fs/data/codebert'
# 加载 CodeBERT 模型和分词器
em_tokenizer = RobertaTokenizer.from_pretrained(em_model_path)
em_model = RobertaModel.from_pretrained(em_model_path)

# 生成器模型
gen_model_path = r'/autodl-fs/data/vicuna-7b-v1.5'

# 加载分词器
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path, use_fast=False)
# 加载模型并分配到设备
gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_path,
    torch_dtype=torch.float16,  # 使用半精度
    device_map="auto"          # 自动分配设备
)

# 创建文本生成 pipeline
generator = pipeline(
    "text-generation",
    model=gen_model,
    tokenizer=gen_tokenizer
)
# 获取代码片段的向量表示
def get_code_embedding(code):
    inputs = em_tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = em_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # 提取 [CLS] 向量
# 拼接检索结果
def get_result():
    input_data_path = r'/autodl-fs/data/dataSet/many_withdiff_largesstubs.json'
    with open(input_data_path, 'r') as f:
        data = json.load(f)
    index = faiss.read_index("code_defect_index.index")
    record_num = 0
    time_sum_start = time.time()
    for record in data:
        # todo 记录每一次生成的时间
        start_time = time.time()
        print("原始数据：", record)
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
        print("添加了检索结果：",record)
        output_template = {
            "bugType": "",
            "defect_code": "",
            "fix": ""
        }
        prompt =f"""
        I will give the following output:{record} . where fixPatch is a code diff information. simpleFix are examples of code fixes similar to the submitted code that you should refer to when answering.
        Your answer should only contain the following information:{output_template}.Among them, bugtype is the code defect type you identified from old_code, defect_code is the specific defect code you identified, and fix means you give the repair code. You just need to give a final output.Please note that you only need to give a json output based on the template I gave, which is {output_template}. Also don’t forget to refer to similarFix.
        """
        # 将生成结果存入文件
        print("prompt：",prompt)
        output = generator(prompt, max_new_tokens=2000, do_sample=True, temperature=0.7)
        # 提取 'generated_text' 字段中的最后一个 JSON
        generated_text = output[0]['generated_text']
        print("原始结果：",generated_text)

        # 查找最后一个 "{'bugType':"
        start_index = generated_text.rfind("following output:{'bugType':")

        # 查找最后一个 "}"
        end_index = generated_text.find("'}") + 1  # 加1是为了包括最后的 '}'  # 加1是为了包括最后的 '}'

        # 获取完整的 JSON 字符串
        json_str = generated_text[start_index:end_index]
        json_str = json_str.replace("'", "\"").replace("\\\"", "\"")
        json_str = json_str.replace("following output:", "", 1)
        end_time = time.time()
        generate_time = round(end_time - start_time, 2)
        json_str += f',"generate_time":{generate_time}'
        # json_data = json.loads(json_str)
        # json_data["generate_time"] = generate_time
        # json_str = json.dumps(json_data)
        if json_str == "":
            if os.path.exists(error_output_file_path) and os.path.getsize(error_output_file_path) > 0:
                with open(error_output_file_path, 'a', encoding='utf-8') as f:
                    f.write(',\n' + json.dumps(record))
            else:
                with open(error_output_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record))
        else:
            if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(',\n' + json_str)
            else:
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(json_str)
        print("生成的结果：",json_str)

        record_num += 1
        if record_num > 1000:
            break
    time_sum_end = time.time()
    time_sum = round(time_sum_end - time_sum_start,2)
    print("该数据集数量：",record_num)
    print("总时间：",time_sum)
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


