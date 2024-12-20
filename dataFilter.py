import json

# 数据筛选：150406条筛选完剩下26727条
# 定义文件路径
file_path = "D:\\GRADUATION\\Nju\\Dataset\\codereviewer\\Code_Refinement\\ref-train.jsonl"

# 定义关键词
keywords = ["fix", "error","bug","issue","mistake","incorrect","fault","defect","flaw","type"]

# 用于存储筛选后的记录
filtered_data = []
lenOr = 0
# 打开并读取 JSON 数据文件
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        # print(line)
        # break
        lenOr += 1
        # 逐行读取每个 JSON 对象
        record = json.loads(line.strip())

        # 获取 "comment" 字段内容
        comment = record.get("comment", "").lower()

        # 检查是否包含关键词
        if any(keyword in comment for keyword in keywords):
            filtered_data.append(record)
            # print(line)
# 输出筛选后的结果
print(f"原始记录数为{lenOr}")
output_path = "D:\\GRADUATION\\Nju\\Dataset\\MyDataSet\\codereviewer_v1\\filtered_data.json"
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(filtered_data, output_file, indent=4, ensure_ascii=False)

print(f"筛选完成，共找到 {len(filtered_data)} 条记录，已保存到 {output_path}")
