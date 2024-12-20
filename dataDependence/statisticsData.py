import json

# 读取 JSON 文件
filename = r'D:\GRADUATION\Nju\Dataset\ManySStuBs4J\testData'
with open(filename, "r", encoding="utf-8") as file:
    data = json.load(file)

# 统计数据条目数量
print(data[:1])  # 打印前两条数据
print(f"数据条目总数: {len(data)}")
