import json
import faiss
import numpy as np
import pickle
from transformers import RobertaTokenizer, RobertaModel, pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import time
from langchain.prompts import PromptTemplate
import re
import pandas as pd

from evaluation.bleu_rouge import *
from evaluation.precision import calculate_precision, calculate_defect_type_precision

import os

TOP_K = 2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
error_output_file_path = 'error_output.json'
output_file_path = 'output.json'

output_train_data_path = r'/autodl-fs/data/testCode/dataDependence/train.jsonl'
print("output_train_data_path:", output_train_data_path)

# 编码器模型
# em_model_path = r'/autodl-fs/data/codebert'
# trained_weights_path = r'/autodl-fs/data/retriver/RAG_FOR_ACR/save/contrastive_model.pt'
# # 加载 CodeBERT 模型和分词器
# em_tokenizer = RobertaTokenizer.from_pretrained(em_model_path)
# em_model = RobertaModel.from_pretrained(em_model_path)
# em_model.load_state_dict(torch.load(trained_weights_path), strict=False)
# em_model.eval()

print("metadata_old_code_cb_v1")
# load_dir = "/autodl-fs/data/testCode/retriver/RAG_FOR_ACR/save4"  # 指定保存模型的路径
load_dir = r'/autodl-fs/data/codebert'
print("load_dir:", load_dir)
print("load_dir:", load_dir)
em_tokenizer = RobertaTokenizer.from_pretrained(load_dir)
em_model = RobertaModel.from_pretrained(load_dir)

# 生成器模型
gen_model_path = r'/autodl-fs/data/vicuna-7b-v1.5'

# 加载分词器
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path, use_fast=False)
# 加载模型并分配到设备
gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_path,
    torch_dtype=torch.float16,  # 使用半精度
    device_map="auto"  # 自动分配设备
)
gen_model.eval()
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
def get_train_data():
    # 输入和输出路径
    input_data_path = r'/autodl-fs/data/dataSet/many_withdiff_largesstubs.json'
    output_train_data_path = r'trainv3_usetestdata.jsonl'  # 训练数据输出路径

    # 读取输入数据
    with open(input_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 读取FAISS索引
    index = faiss.read_index("mr_cbforsure.index")

    # 读取元数据（加载一次，避免在循环中重复读取）
    with open("metadata_mr_cbforsure.pkl", "rb") as f:
        metadata = pickle.load(f)

    renum_now = 0
    records_processed = 0

    # 打开输出文件一次，避免在循环中频繁打开
    with open(output_train_data_path, 'w', encoding='utf-8') as out_f:
        for record in data:
            renum_now += 1

            # 跳过前20000条记录
            # if renum_now < 20000:
            #     continue

            print(f"Processing record number: {renum_now}")

            # 创建记录副本并移除不需要的键
            record_remove_key = record.copy()
            keys_to_remove = [
                'sourceAfterFix', 'sourceBeforeFix', 'fixCommitSHA1', 'fixCommitParentSHA1',
                'bugFilePath', 'fixPatch', 'bugLineNum', 'projectName', 'bugNodeStartChar',
                'bugNodeLength', 'fixLineNum', 'fixNodeStartChar', 'fixNodeLength', 'new_code'
            ]

            for key in keys_to_remove:
                record_remove_key.pop(key, None)  # 使用 pop，若键不存在则返回 None，不会抛出错误

            # print("Record after removing keys:", record_remove_key)

            start_time = time.time()

            # 获取原始代码
            old_code = record.get('old_code', '')
            # print("Old Code:", old_code)

            # 获取代码的向量表示
            query_vector = get_code_embedding(record.get('sourceBeforeFix', ''))
            query_vector = np.array([query_vector], dtype=np.float32)

            # 归一化查询向量
            faiss.normalize_L2(query_vector)

            # 进行检索，返回最相似的k个结果
            k = 2  # TOP_K
            D, I = index.search(query_vector, k)

            with open("metadata_mr_cbforsure.pkl", "rb") as f:
                metadata = pickle.load(f)
            result_record = []
            for idx in I[0]:
                result_record.append(metadata[idx])
            record_remove_key['similarFix'] = result_record
            # print("检索结果：", result_record)
            output_template = {
                "bugtype_out": "CHANGE_CALLER_IN_FUNCTION_CALL",
                "defect_code": "stacktrace.indexOf(':')",
                "fix": "firstLine.indexOf(':')"
            }
            # 构建prompt
            prompt = (
        f"""
        You are a coding assistant.Analyze the old_code I gave you, and use similarFix as a reference.\n
            \t- First, identify the defect_code in old_code by analyzing it in comparison with the similarFix examples.\n
            \t- Second, generate a fix for the identified defect_code.\n
            \t- Third, classify the defect into one of the predefined bug types.\n
            ** Bug Types **\n
            Use the following bug types to classify any identified defect:\n
            'OVERLOAD_METHOD_MORE_ARGS', 'SWAP_BOOLEAN_LITERAL', 'CHANGE_MODIFIER', 'CHANGE_NUMERAL', 'CHANGE_OPERAND', 'ADD_THROWS_EXCEPTION', 'DELETE_THROWS_EXCEPTION', 'OVERLOAD_METHOD_DELETED_ARGS', 'CHANGE_CALLER_IN_FUNCTION_CALL', 'MORE_SPECIFIC_IF', 'CHANGE_OPERATOR', 'CHANGE_IDENTIFIER', 'DIFFERENT_METHOD_SAME_ARGS', 'CHANGE_UNARY_OPERATOR', 'SWAP_ARGUMENTS', 'LESS_SPECIFIC_IF', 'CHANGE_NUMERAL', 'MORE_SPECIFIC_IF', 'CHANGE_UNARY_OPERATOR', 'CHANGE_OPERATOR', 'OVERLOAD_METHOD_DELETED_ARGS', 'CHANGE_CALLER_IN_FUNCTION_CALL', 'OVERLOAD_METHOD_MORE_ARGS', 'ADD_THROWS_EXCEPTION', 'SWAP_ARGUMENTS', 'CHANGE_MODIFIER', 'LESS_SPECIFIC_IF', 'CHANGE_OPERAND', 'DIFFERENT_METHOD_SAME_ARGS', 'CHANGE_IDENTIFIER', 'DELETE_THROWS_EXCEPTION', 'SWAP_BOOLEAN_LITERAL'\n\n

            Input:\n
                old_code: {old_code}\n
                similarFix: {record_remove_key['similarFix']}\n
        Standard output Example:{output_template}defect_code is the defective code you found from sourceBeforeFix, and fix is the repair code you gave for the identified defect_code. Please note that this is just a template, and the code and other information inside is just an example, not what you need to refer to.\n
        output must be valid JSON.Your final output should only have one result.And you should start with 'final output' before final output.
        
            """
            )
            print("Prompt:", prompt)
            # prompt = (
            #     f"{instruction}\n\n"
            #     f"OLD CODE:\n{old_code}\n\n"
            #     f"SIMILAR FIX EXAMPLES:\n{similar_fix_text}"
            #     f"QUESTION:\nPlease output the following three fields: defect_code, bugType, fix.\n\n"
            #     f"ANSWER:\n"
            # )

            # 构建response
            response = {
                "defect_code": record.get('sourceBeforeFix', ''),
                "bugType": record.get('bugType', ''),
                "fix": record.get('sourceAfterFix', '')
            }
            # 将response转换为JSON字符串
            response_json = json.dumps(response, ensure_ascii=False)
            print("Response:", response_json)
            # break
            response ={
                "final output:\n",
                    response_json
            }
            # 构建最终的训练样本
            train_sample = {
                "prompt": prompt,
                "response": response_json
            }

            print("Training Sample:", train_sample)

            # 写入到JSON Lines文件
            out_f.write(json.dumps(train_sample, ensure_ascii=False) + '\n')

            end_time = time.time()
            generate_time = round(end_time - start_time, 2)
            print(f"Record processed in {generate_time} seconds")

            records_processed += 1

            # 设置处理的记录数量上限（例如100条），根据需要调整或移除
            if records_processed >= 1000:
                break

    print(f"Total records processed: {records_processed}")


if __name__ == '__main__':
    get_train_data()

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


