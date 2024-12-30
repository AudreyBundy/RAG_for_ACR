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
from evaluation.precision import calculate_precision,calculate_defect_type_precision


import os

TOP_K = 2



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
error_output_file_path = 'error_output.json'
output_file_path = 'output.json'

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
print("load_dir:",load_dir)
print("load_dir:",load_dir)
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
    device_map="auto"          # 自动分配设备
)
gen_model.eval()
# 创建文本生成 pipeline
generator = pipeline(
    "text-generation",
    model=gen_model,
    tokenizer=gen_tokenizer
)

# 从模型输出中提取结果
def output_split(generated_text):
    # 找到起始点的文本
    marker = "output must be valid JSON.Your final output should only have one result.And you should start with 'final output' before final output"

    # 提取 marker 后的内容
    marker_index = generated_text.find(marker)
    if marker_index != -1:
        # 只分析 marker 后的部分
        generated_text = generated_text[marker_index + len(marker):]
        # print(generated_text)
    else:
        # 如果没有找到 marker，返回 None
        return None, None, None, None

    def extract_first_match(pattern, text):
        """
        使用正则表达式提取第一个匹配项。
        """
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    # 正则模式
    patterns = {
        "has_issue": r"['\"]has\\?_issue['\"]:\s*(true|false|null|['\"](.*?)['\"])",
        "bug_type": r"['\"]bugtype\\?_out['\"]:\s*['\"](.*?)['\"]",
        "defect_code": r"['\"]defect\\?_code['\"]:\s*([^\n]+)",
        "fix": r"['\"]fix['\"]:\s*([^\n]+)"
    }
    def clean_value(value):
        """
        对提取的值进行清理：
        1. 删除第一个字符。
        2. 如果最后一个字符是逗号，则删除最后两个字符。
        3. 否则，删除最后一个字符。
        """
        if value:
            # 如果第一个字符是单引号或双引号，则删除第一个字符
            if value[0] in ("'", '"'):
                value = value[1:]
            # 删除最后一个或两个字符
            if value.endswith(','):
                value = value[:-2]
            elif value.endswith(("'", '"')):
                value = value[:-1]
            # 去除可能的多余空白字符
            return value.strip()
        return value
    def truncate_before_fix(value):
        """
        如果 value 中包含 "fix" 或 'fix'，则截断到 "fix" 或 'fix' 前边。
        """
        if not value:
            return value

        # 使用正则表达式查找 "fix" 或 'fix'，忽略大小写
        match = re.search(r'["\']?fix["\']?', value, re.IGNORECASE)
        if match:
            # 截断到 "fix" 或 'fix' 前边
            return value[:match.start()].strip()
        return value
    def truncate_before_similar(value):
        """
        如果 value 中包含 "fix" 或 'fix'，则截断到 "fix" 或 'fix' 前边。
        """
        if not value:
            return value

        # 使用正则表达式查找 "fix" 或 'fix'，忽略大小写
        match = re.search(r'["\']?similarFix["\']?', value, re.IGNORECASE)
        if match:
            # 截断到 "fix" 或 'fix' 前边
            return value[:match.start()].strip()
        return value
    # 提取字段
    hasissue_value = extract_first_match(patterns["has_issue"], generated_text)
    bug_type_value = extract_first_match(patterns["bug_type"], generated_text)
    defect_code_value = extract_first_match(patterns["defect_code"], generated_text)
    fix_value = extract_first_match(patterns["fix"], generated_text)
    # 清理提取的值
    defect_code_value = clean_value(defect_code_value)
    fix_value = clean_value(fix_value)
    # 防止出现输出都在同一行的情况
    defect_code_value = truncate_before_fix(defect_code_value)
    defect_code_value = clean_value(defect_code_value)
    fix_value = truncate_before_similar(fix_value)
    fix_value = clean_value(fix_value)

    # print(hasissue_value)
    # print(bug_type_value)
    print(defect_code_value)
    print(fix_value)

    return hasissue_value, bug_type_value, defect_code_value, fix_value


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
    # / autodl - fs / data / dataSet
    input_data_path = r'/autodl-fs/data/dataSet/many_withdiff_largesstubs.json'
    with open(input_data_path, 'r') as f:
        data = json.load(f)
    index = faiss.read_index("dataDependence/mr_cbforsure.index")
    record_num = 0
    true_num = 0
    time_sum_start = time.time()
    for record in data:
        # todo 原始数据删除结果
        record_remove_key = record.copy()
        keys_to_remove = ['bugType',  'sourceAfterFix','sourceBeforeFix', 'fixCommitSHA1', 'fixCommitParentSHA1',
                          'bugFilePath', 'fixPatch', 'bugLineNum', 'projectName', 'bugNodeStartChar',
                          'bugNodeLength', 'fixLineNum', 'fixNodeStartChar', 'fixNodeLength', 'new_code']

        for key in keys_to_remove:
            record_remove_key.pop(key, None)  # 使用 pop，若键不存在则返回 None，不会抛出错误
        print("record:",record_remove_key)
        start_time = time.time()
        print("原始数据：", record_remove_key)
        # 是否添加rag
        change_code = record.get('sourceBeforeFix', '')
        print("新代码：",change_code)
        query_vector = get_code_embedding(change_code)
        query_vector = np.array([query_vector], dtype=np.float32)
        # 归一化查询向量
        # todo 对比方法记得换
        faiss.normalize_L2(query_vector)
        # 进行检索，返回最相似的k个结果
        D, I = index.search(query_vector, k=2)  # 返回最相似的两个结果
        # 载入元数据
        with open("dataDependence/metadata_mr_cbforsure.pkl", "rb") as f:
            metadata = pickle.load(f)
        result_record = []
        for idx in I[0]:
            result_record.append(metadata[idx])
        record_remove_key['similarFix'] = result_record
        print("检索结果：",result_record)
        output_template = {
            "has_issue": "True",
            "bugtype_out": "CHANGE_CALLER_IN_FUNCTION_CALL",
            "defect_code": "stacktrace.indexOf(':')",
            "fix": "firstLine.indexOf(':')"
        }
        print("record:",record_remove_key)
        # todo 类别给个全集，
        # prompt = f"""
        # You are a coding assistant. Your task is to analyze the given record and produce a JSON output.
        # description: You are a coding assistant tasked with analyzing and fixing code defects. Your job is to:
        # 1. Analyze the `old_code` provided in the input.
        # 2. Identify if there is a defect:
        #     - If no defect exists, output:  "has_issue": "False"
        #     and stop.
        # 3. If a defect exists:
        # - Determine the **bugType** from the given list of bug types.
        # - Provide the **location of the defect** in `old_code`.
        # - Generate a **code review comment** to explain the defect and the reasoning behind the suggested fix.
        # - Suggest a repair for the identified defect.
        #
        #
        # ### **Bug Types**
        # Use the following bug types to classify any identified defect:
        # 'OVERLOAD_METHOD_MORE_ARGS', 'SWAP_BOOLEAN_LITERAL', 'CHANGE_MODIFIER', 'CHANGE_NUMERAL', 'CHANGE_OPERAND', 'ADD_THROWS_EXCEPTION', 'DELETE_THROWS_EXCEPTION', 'OVERLOAD_METHOD_DELETED_ARGS', 'CHANGE_CALLER_IN_FUNCTION_CALL', 'MORE_SPECIFIC_IF', 'CHANGE_OPERATOR', 'CHANGE_IDENTIFIER', 'DIFFERENT_METHOD_SAME_ARGS', 'CHANGE_UNARY_OPERATOR', 'SWAP_ARGUMENTS', 'LESS_SPECIFIC_IF','CHANGE_NUMERAL', 'MORE_SPECIFIC_IF', 'CHANGE_UNARY_OPERATOR', 'CHANGE_OPERATOR', 'OVERLOAD_METHOD_DELETED_ARGS', 'CHANGE_CALLER_IN_FUNCTION_CALL', 'OVERLOAD_METHOD_MORE_ARGS', 'ADD_THROWS_EXCEPTION', 'SWAP_ARGUMENTS', 'CHANGE_MODIFIER', 'LESS_SPECIFIC_IF', 'CHANGE_OPERAND', 'DIFFERENT_METHOD_SAME_ARGS', 'CHANGE_IDENTIFIER', 'DELETE_THROWS_EXCEPTION', 'SWAP_BOOLEAN_LITERAL'
        #
        #
        # ### **Input Description**
        # - 'old_code': The source code that may or may not contain a defect.
        # Input:
        # {record_remove_key}
        # Standard output Example:
        # {output_template}
        # Output must be valid JSON.Your final output should only have one result.And you should start with 'final output' before final output.The final output should be given by you.
        # """
        # ### **Input Description**
        # - 'old_code': The source code that may or may not contain a defect.
        # - `similarFix`: A list of reference bug fixes or examples. Each entry includes:
        #     - `has_issue`: `"True"` for a defect fix example or `"False"` for a no-issue example.
        #     - `bugType`: The type of defect (if applicable).
        #     - `old`: The defective code snippet (for `"True"`) or a clean code snippet (for `"False"`).
        #     - `new`: The corrected code snippet (for `"True"`).


        prompt =f"""
        You are a coding assistant. Analyze the old_code I gave you, and use similarFix as a reference.
            - First, identify the defect_code in old_code by analyzing it in comparison with the similarFix examples.
            - Second, generate a fix for the identified defect_code.
            - Third, classify the defect into one of the predefined bug types.
        Input:
                old_code: {record_remove_key['old_code']}
                similarFix: {record_remove_key['similarFix']}{record_remove_key}
        Standard output Example:{output_template}defect_code is the defective code you found from sourceBeforeFix, and fix is the repair code you gave for the identified defect_code. Please note that this is just a template, and the code and other information inside is just an example, not what you need to refer to.
        output must be valid JSON.Your final output should only have one result.And you should start with 'final output' before final output.
        """
        # prompt = f"""
        # You are a coding assistant. Analyze the old_code I gave you, and use similarFix as a reference.
        # - First, identify the defect_code in old_code by analyzing it in comparison with the similarFix examples.
        # - Second, generate a fix for the identified defect_code.
        # - Third, classify the defect into one of the predefined bug types.
        # Input:
        # old_code: {record_remove_key['old_code']}
        # similarFix: {record_remove_key['similarFix']}
        #
        # Output (JSON format):
        # - has_issue: "True" or "False"
        # - defect_code: The part of old_code that contains the defect.
        # - fix: Your fix for the defect_code.
        # - bugType: The type of the defect (use predefined types).
        #
        # ###Output Example:
        # {{
        #   "has_issue": "True",
        #   "defect_code": "stacktrace.indexOf(':')",
        #   "fix": "firstLine.indexOf(':')",
        #   "bugType": "CHANGE_CALLER_IN_FUNCTION_CALL"
        # }}
        # Output must be valid JSON.And you should output a unique final output
        # """

        t = "Note that simpleFix is an example of a fix that you need to refer to, but not your output."
        # 将生成结果存入文件
        # prompt = PromptTemplate(
        #     input_variables=["record", "output_template"],
        #     template=prompt
        # )
        # print("prompt：",prompt)
        output = generator(prompt, max_new_tokens=4096, do_sample=True, temperature=0.7)
        # 提取 'generated_text' 字段中的最后一个 JSON
        generated_text = output[0]['generated_text']
        print("原始结果：",generated_text)

        hasissue,bugtype,defect_code,fix = output_split(generated_text)
        if hasissue is None or bugtype is None or defect_code is None or fix is None:
            continue
        if bugtype:
            bugtype = bugtype.replace("\\","")
        if defect_code:
            defect_code = defect_code.replace("\\","")
        if fix:
            fix = fix.replace("\\", "")
        # print("缺陷类型：",bugtype)
        # print("缺陷代码",defect_code)
        # print("修复代码",fix)
        end_time = time.time()
        generate_time = round(end_time - start_time, 2)
        # print("生成时间：",generate_time)

        # 处理fix_code，方便计算bleu
        # dealedfix = deal_fix_code(defect_code,record.get('sourceBeforeFix', ''),fix,record.get('sourceAfterFix', ''))
        # bleu_per_new = calculate_bleu(fix, dealedfix)


        # 计算BLEU和ROUGE
        # todo bleu使用两个，
        bleu_per = calculate_bleu(fix, record.get('sourceAfterFix', ''))
        rouge1_per,rouge2_per,rougeL_per = calculate_rouge(fix, record.get('sourceAfterFix', ''))
        precision_pre = calculate_precision(defect_code, record.get('sourceBeforeFix', ''))
        precision_bugtype_pre = calculate_defect_type_precision(bugtype, record.get('bugType', ''))

        # bleu_per = max(bleu_per,bleu_per_new)

        bleu += bleu_per
        rouge1 += rouge1_per
        rouge2 += rouge2_per
        rougeL += rougeL_per
        precision += precision_pre


        # 创建或更新 DataFrame
        data = {
            "有无缺陷":[hasissue],
            "缺陷类型": [bugtype],
            "原始缺陷类型": [record.get('bugType', '')],
            "缺陷代码": [defect_code],
            "原始缺陷代码": [record.get('sourceBeforeFix', '')],
            "修复代码": [fix],
            "原始修复代码": [record.get('sourceAfterFix', '')],
            "单次生成时间": [generate_time],
            "本次生成的BLEU": [bleu_per],
            "本次生成的rouge1": [rouge1_per],
            "本次生成的rouge2": [rouge2_per],
            "本次生成的rougeL": [rougeL_per],
            "本次识别缺陷类型的precision": [precision_pre],
            "本次识别位置的precision": [precision_bugtype_pre],
            "fixCommitSHA1": [record.get('fixCommitSHA1', '')],
            "fixCommitParentSHA1": [record.get('fixCommitParentSHA1', '')],
        }
        print("生成的数据：",data)
        # 定义文件路径
        # file_path = "/root/autodl-fs/dataSet/output/gen_data_myp_mr_truere_v1.xlsx"
        file_path = "/root/autodl-fs/dataSet/output/mr_cbforsure.xlsx"
        print("file_path:",file_path)
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

        if hasissue == "True" or hasissue == "true":
            true_num += 1
            print("true 的数据量：",true_num)
        record_num += 1
        if record_num > 100:
            break
    time_sum_end = time.time()
    time_sum = round(time_sum_end - time_sum_start,2)

    bleu = round(bleu / record_num, 4)
    rouge1 = round(rouge1 / record_num, 4)
    rouge2 = round(rouge2 / record_num, 4)
    rougeL = round(rougeL / record_num, 4)
    precision = round(precision / record_num, 4)
    print("本次运行数据量：",record_num)
    print("true 的数据量：",true_num)
    print("总时间：",time_sum)
    data = {
        "缺陷类型": 0,
        "原始缺陷类型": [record.get('bugType', '')],
        "缺陷代码": 0,
        "修复代码": 0,
        "单次生成时间": [time_sum],
        "本次生成的BLEU": [bleu],
        "本次生成的rouge1": [rouge1],
        "本次生成的rouge2": [rouge2],
        "本次生成的rougeL": [rougeL],
        "本次识别缺陷类型的precision": [precision_pre],
        "本次识别的precision": [precision],
        "fixCommitSHA1": 0,
        "fixCommitParentSHA1": 0,
    }
    print("生成的数据：", data)
    # 定义文件路径
    file_path = "/root/autodl-fs/dataSet/output/mr_cbforsure.xlsx"

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


