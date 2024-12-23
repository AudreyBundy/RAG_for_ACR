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

from evaluation.bleu_rouge import calculate_bleu, calculate_rouge
from evaluation.precision import calculate_precision


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

# 从模型输出中提取结果
def output_split(generated_text):


    # 使用正则表达式寻找最后一个'bugType':或者"bugType":及其后的内容
    bugtype = list(
        re.finditer(r"['\"]bugtype\\?_out['\"]:\s*['\"](.*?)['\"]", generated_text, re.IGNORECASE)
    )
    # 提取最后一个匹配结果
    if bugtype:
        last_match = bugtype[-1]
        bug_type_value = last_match.group(1)
    else:
        bug_type_value = None

    # 使用正则表达式寻找最后一个'bugType':或者"bugType":及其后的内容
    defect_code = list(
        re.finditer(r"['\"]defect\\?_code['\"]:\s*['\"](.*?)['\"]", generated_text, re.IGNORECASE)
    )
    # 提取最后一个匹配结果
    if defect_code:
        last_match = defect_code[-1]
        defect_code_value = last_match.group(1)
    else:
        defect_code_value = None

    # 使用正则表达式寻找最后一个'bugType':或者"bugType":及其后的内容
    fix = list(
        re.finditer(r"['\"]fix['\"]:\s*['\"](.*?)['\"]", generated_text, re.IGNORECASE)
    )
    # 提取最后一个匹配结果
    if fix:
        last_match = fix[-1]
        fix_value = last_match.group(1)
    else:
        fix_value = None

    return bug_type_value, defect_code_value, fix_value


# 获取代码片段的向量表示
def get_code_embedding(code):
    inputs = em_tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = em_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()  # 提取 [CLS] 向量
# 拼接检索结果
def get_result():
    # 评估指标
    bleu = 0.0
    rouge1 = 0.0
    rouge2 = 0.0
    rougeL = 0.0
    precision = 0.0

    input_data_path = r'/autodl-fs/data/dataSet/many_withdiff_largesstubs.json'
    with open(input_data_path, 'r') as f:
        data = json.load(f)
    index = faiss.read_index("code_defect_index.index")
    record_num = 0
    time_sum_start = time.time()
    for record in data:
        start_time = time.time()
        # print("原始数据：", record)
        new_code = record.get('new', '')
        query_vector = get_code_embedding(new_code)
        query_vector = np.array([query_vector], dtype=np.float32)
        # 归一化查询向量
        # todo 对比方法记得换
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
        # print("添加了检索结果：",record)
        output_template = {
            "bugtype_out": "CHANGE_CALLER_IN_FUNCTION_CALL",
            "defect_code": "stacktrace.indexOf(':')",
            "fix": "firstLine.indexOf(':')"
        }
        prompt =f"""
        You are a coding assistant. Your task is to analyze the given record and produce a JSON output.
        description: You are a coding assistant.Your task is to analyze the code diff I gave you, identify the only defective code in it, and give the defect type, as well as your suggested repair code.Refer to simpleFix while generating the fix.
        Input: {record}
        standard output Example:{output_template}
        Output must be valid JSON.Your final output should only have one result.And you should start with 'final output' before final output
        """
        # 将生成结果存入文件
        # prompt = PromptTemplate(
        #     input_variables=["record", "output_template"],
        #     template=prompt
        # )
        # print("prompt：",prompt)
        output = generator(prompt, max_new_tokens=2000, do_sample=True, temperature=0.7)
        # 提取 'generated_text' 字段中的最后一个 JSON
        generated_text = output[0]['generated_text']
        print("原始结果：",generated_text)

        bugtype,defect_code,fix = output_split(generated_text)

        bugtype = bugtype.replace("\\","")
        # print("缺陷类型：",bugtype)
        # print("缺陷代码",defect_code)
        # print("修复代码",fix)
        end_time = time.time()
        generate_time = round(end_time - start_time, 2)
        # print("生成时间：",generate_time)

        # 计算BLEU和ROUGE
        bleu_per = calculate_bleu(fix, record.get('sourceAfterFix', ''))
        rouge1_per,rouge2_per,rougeL_per = calculate_rouge(fix, record.get('sourceAfterFix', ''))
        precision_pre = calculate_precision(defect_code, record.get('sourceBeforeFix', ''))

        bleu += bleu_per
        rouge1 += rouge1_per
        rouge2 += rouge2_per
        rougeL += rougeL_per
        precision += precision_pre


        # 创建或更新 DataFrame
        data = {
            "缺陷类型": [bugtype],
            "缺陷代码": [defect_code],
            "修复代码": [fix],
            "单次生成时间": [generate_time],
            "本次生成的BLEU": [bleu_per],
            "本次生成的rouge1": [rouge1_per],
            "本次生成的rouge2": [rouge2_per],
            "本次生成的rougeL": [rougeL_per],
            "本次识别的precision": [precision_pre],
            "fixCommitSHA1": [record.get('fixCommitSHA1', '')],
            "fixCommitParentSHA1": [record.get('fixCommitParentSHA1', '')],
        }
        print("生成的数据：",data)
        # 定义文件路径
        file_path = "/root/autodl-fs/dataSet/output/gen_data_v2.xlsx"

        # 将新数据转换为 DataFrame
        new_data = pd.DataFrame(data)

        try:
            # 尝试加载现有文件
            with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                # 写入新数据到同一工作表的最后一行
                new_data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        except FileNotFoundError:
            # 如果文件不存在，直接创建新文件
            new_data.to_excel(file_path, index=False)
        record_num += 1
        if record_num > 2:
            break
    time_sum_end = time.time()
    time_sum = round(time_sum_end - time_sum_start,2)

    bleu = round(bleu / record_num, 4)
    rouge1 = round(rouge1 / record_num, 4)
    rouge2 = round(rouge2 / record_num, 4)
    rougeL = round(rougeL / record_num, 4)
    precision = round(precision / record_num, 4)
    print("本次运行数据量：",record_num)
    print("总时间：",time_sum)
    data = {
        "缺陷类型": 0,
        "缺陷代码": 0,
        "修复代码": 0,
        "单次生成时间": [time_sum],
        "本次生成的BLEU": [bleu],
        "本次生成的rouge1": [rouge1],
        "本次生成的rouge2": [rouge2],
        "本次生成的rougeL": [rougeL],
        "本次识别的precision": [precision],
        "fixCommitSHA1": 0,
        "fixCommitParentSHA1": 0,
    }
    print("生成的数据：", data)
    # 定义文件路径
    file_path = "/root/autodl-fs/dataSet/output/gen_data_v2.xlsx"

    # 将新数据转换为 DataFrame
    new_data = pd.DataFrame(data)

    try:
        # 尝试加载现有文件
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            # 写入新数据到同一工作表的最后一行
            new_data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    except FileNotFoundError:
        # 如果文件不存在，直接创建新文件
        new_data.to_excel(file_path, index=False)
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


