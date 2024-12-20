import os
import json
import subprocess
import re

updated_data = []

# 处理每个 JSON 数据的函数
def process_json_data(json_data, output_dir, lang_filter="py"):
    file_count = 1  # 用于生成不同的文件名
    for entry in json_data:
        # 获取语言类型
        lang = entry.get("lang", "").lower()

        if lang == lang_filter:  # 只处理指定语言
            # 获取代码片段
            oldf = entry.get("oldf", "")
            old_hunk = entry.get("old_hunk", "")
            modify_lines = extract_line_numbers_from_hunk(oldf, old_hunk)
            print(modify_lines)
            # break
            if oldf:
                # 解析转义字符，如 \n 转换为换行符
                oldf = oldf.encode().decode('unicode_escape')
                update_data = []
                # 创建文件夹
                lang_dir = os.path.join(output_dir, lang)
                os.makedirs(lang_dir, exist_ok=True)

                # 生成文件路径并写入代码
                file_name = os.path.join(lang_dir, f"code{file_count}.py")
                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write(oldf)
                print(f"Code written to {file_name}")
                # 确定是否有缺陷
                errors = extract_and_analyze(file_name)
                for error in errors:
                    if modify_lines[0] <= int(error['line']) <= modify_lines[1]:
                        print(f"Defect found in line {error['line']}: {error['error']}")
                        entry["defects"].append(error['error'])
                        update_data.append(entry)
                        break
                if update_data:
                    updated_data.append(update_data)

                # 更新文件计数
                file_count += 1
                if file_count > 4:
                    break

# 从指定文件读取数据
def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 提取被修改内容范围
def extract_line_numbers_from_hunk(oldf, old_hunk):
    # 匹配旧文件行号和修改行数
    match = re.match(r"@@ -(\d+),(\d+) \+\d+,\d+ @@", old_hunk)
    if not match:
        raise ValueError("Invalid hunk header format")

    start_line = int(match.group(1))  # 起始行号
    num_lines = int(match.group(2))  # 修改的行数
    end_line = start_line + num_lines - 1  # 结束行号
    modify_lines = [start_line, end_line]
    # 生成旧文件中的被修改行号范围
    return modify_lines

# 解析 Pytype 输出
def parse_pytype_output(pytype_stdout):
    errors = []
    lines = pytype_stdout.splitlines()
    for line in lines:
        print(line)
        line = str(line)
        pattern = r'(\d+):\d+: error:.*\[(\w+-\w+)\]'
        match = re.search(pattern, line)
        print(match)
        if match:
            # file_path = match.group(1)
            line_number = match.group(1)  # 提取行号
            defect_type = match.group(2)  # 提取缺陷类型
            errors.append({'line': line_number, 'error': defect_type})
    return errors


# 使用静态工具分析代码。获取缺陷
def extract_and_analyze(file_name):
    result = subprocess.run(['pytype', file_name], capture_output=True, text=True)
    errors = parse_pytype_output(result.stdout)
    print(errors)
    return errors
# 主程序
def main():
    # 指定 JSON 文件路径
    json_file_path = r"D:\GRADUATION\Nju\Dataset\MyDataSet\codereviewer_v1\filtered_data.json"

    # 读取 JSON 数据
    json_data = load_data_from_json(json_file_path)

    # 设置输出目录
    output_dir = r"D:\GRADUATION\Nju\Dataset\MyDataSet\codereviewer_v1\javaCode_v1"

    # 处理  代码
    process_json_data(json_data, output_dir, lang_filter="py")
    # 创建新的数据
    update_fine_loc = "D:\\GRADUATION\\Nju\\Dataset\\MyDataSet\\codereviewer_v1\\filtered_data.json"
    with open(update_fine_loc, "w", encoding="utf-8") as output_file:
        json.dump(updated_data, output_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
    # result = subprocess.run(['pytype', 'D:\\GRADUATION\\Nju\\Dataset\\MyDataSet\\codereviewer_v1\\javaCode_v1\\py\\code2.py'], capture_output=True, text=True)
    # print(result.stdout)
    # extract_and_analyze('D:\\GRADUATION\\Nju\\Dataset\\MyDataSet\\codereviewer_v1\\javaCode_v1\\py\\code2.py')
