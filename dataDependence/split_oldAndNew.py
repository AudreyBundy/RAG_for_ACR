import json
import re


file_path = r'/autodl-fs/data/dataSet/pyData/split_data/test.json'
output_path = "/autodl-fs/data/dataSet/pyData/split_data/test2.json"
error_data_output_path = "D:\\GRADUATION\\Nju\\Dataset\\MyDataSet\\codereviewer_v1\\eror_data.json"
# 存储新的数据，添加了old_code和new_code
modify_data = []
# 问题数据
error_data = []
# 读取 JS
# ON 数据并处理
with open(file_path, 'r',encoding='utf-8') as f:
    data_list = json.load(f)


def get_diff_and_save():
    # 统计写入的数据条
    i = 0
    error_num = 0
    change_block_regex = r'@@\s?-\d+,\d+\s?\+\d+,\d+\s?@@([\s\S]*?)(?=@@|\Z)'
    for data in data_list:
        # 分别代表修改前和修改后的代码
        block_without_new_code = ""
        block_without_old_code = ""

        code_diff = data["diff"]
        source_before_fix = data["before"]
        print("处理之前的codediff:", code_diff)
        print("修改的地方:", source_before_fix)
        change_blocks = re.findall(change_block_regex, code_diff, re.MULTILINE)
        flag = 0
        for block in change_blocks:
            print("处理之前的block:", block)
            block_for_compare = block
            block_for_compare = re.sub(r'^[+-]', '', block_for_compare, flags=re.MULTILINE)
            # source_before_fix_without_space = source_before_fix.replace(" ", "").replace("\n", "")
            # block_without_space = block_for_compare.replace(" ", "").replace("\n", "")
            if source_before_fix.replace(" ", "") in block.replace(" ", ""):
                # 提取修改前的代码行,这是只获取变更的那一行
                # old_code = re.findall(old_code_regex, code_diff, re.MULTILINE)
                #
                # # 提取修改后的代码行
                # new_code = re.findall(new_code_regex, code_diff, re.MULTILINE)
                # 这个是获取整个codediff的变更行
                block_without_new_code = re.sub(r'^\+.*$', '', block, flags=re.MULTILINE)  # 删除修改后的行
                block_without_old_code = re.sub(r'^\-.*$', '', block, flags=re.MULTILINE)  # 删除修改前的行
                # 去掉所有 "+" 和 "-" 符号（保留代码内容）
                block_without_new_code = re.sub(r'^[+-]', '', block_without_new_code, flags=re.MULTILINE)
                block_without_old_code = re.sub(r'^[+-]', '', block_without_old_code, flags=re.MULTILINE)
                print("处理之后的oldcode:", block_without_new_code)
                print("处理之后的newcode:", block_without_old_code)
                data["old_code"] = block_without_new_code
                data["new_code"] = block_without_old_code
                print("要写入的数据:", data)
                i += 1
                modify_data.append(data)
                flag = 1
                break
        # 有些数据未被加入，这里将其加入error_data
        if flag != 1:
            data["old_code"] = data["before"]
            data["new_code"] = data["after"]
            modify_data.append(data)
            error_num += 1
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(modify_data, f, ensure_ascii=False, indent=4)
        print("处理完成，已保存到", output_path)
        print("写入了", i, "条数据")
    print("问题数据数目：", error_num)
    # with open(error_data_output_path, 'w', encoding='utf-8') as f:
    #     json.dump(error_data, f, ensure_ascii=False, indent=4)
    #     print("问题数据数目：", error_num)

if __name__ == "__main__":
    get_diff_and_save()
    print("原始数据长度", len(data_list))