# import json
#
# # 读取 JSON 文件
# filename = r'D:\GRADUATION\Nju\Dataset\ManySStuBs4J\testData'
# with open(filename, "r", encoding="utf-8") as file:
#     data = json.load(file)
#
# # 统计数据条目数量
# print(data[:1])  # 打印前两条数据
# print(f"数据条目总数: {len(data)}")

import json

# 定义一个函数来加载 JSONL 文件
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 定义一个函数来统计并格式化 sstub_pattern
def get_sstub_patterns(data):
    patterns = set()  # 使用集合来避免重复
    for item in data:
        pattern = item.get("sstub_pattern")
        if pattern:
            patterns.add(pattern)
    return patterns

# 加载数据
file_path = '/autodl-fs/data/dataSet/pyData/likely_bug_true.jsonl'  # 替换为你的文件路径
data = load_jsonl(file_path)

# 获取所有 sstub_pattern
patterns = get_sstub_patterns(data)

# 打印格式化后的结果
formatted_patterns = ', '.join(f"'{pattern}'" for pattern in patterns)
print(formatted_patterns)
