# import json
#
# # 假设数据集是一个 JSON 数组，保存在文件 data.json 中
# data_file = r'D:\GRADUATION\Nju\Dataset\ManySStuBs4J\sstubs'
# data_files = ['D:\\GRADUATION\\Nju\\Dataset\\ManySStuBs4J\\sstubs','D:\\GRADUATION\\Nju\\Dataset\\ManySStuBs4J\\sstubsLarge',]
# # 读取 JSON 数据
# with open(data_file, 'r', encoding='utf-8') as file:
#     dataset = json.load(file)
#
# # 使用集合来统计唯一的 bugType
# bug_types = set()
#
# for item in dataset:
#     bug_types.add(item['bugType'])
#
# # 输出结果
# print(f"数据集中一共有 {len(bug_types)} 种不同的 bugType：")
# print(bug_types)
import json

# 假设数据集是一个 JSON 数组，保存在文件 data.json 中
data_file = r'D:\GRADUATION\Nju\Dataset\ssc_data_28M\file-0.jsonl\file-0.jsonl'

# 读取 JSON 数据
with open(data_file, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# 使用集合统计唯一的 sstub_pattern，前提是 likely_bug 为 True
sstub_patterns = {item['sstub_pattern'] for item in dataset if item['likely_bug']}

# 转换为列表并排序（可选）
sstub_patterns_list = sorted(list(sstub_patterns))

# 输出结果
print(f"数据集中在 likely_bug 为 True 的情况下，有 {len(sstub_patterns_list)} 种不同的 sstub_pattern：")
print(sstub_patterns_list)
[    'add_attribute_access',
    'add_elements_to_iterable',
    'add_function_around_expression',
    'add_method_call',
    'add_throws_exception',
    'change_attribute_used',
    'change_binary_operand',
    'change_binary_operator',
    'change_boolean_literal',
    'change_constant_type',
    'change_identifier_used',
    'change_keyword_argument_used',
    'change_numeric_literal',
    'change_unary_operator',
    'change_caller_in_function_call',
    'change_modifier',
    'change_numeral',
    'change_operand',
    'change_operator',
    'delete_throws_exception',
    'different_method_same_args',
    'less_specific_if',
    'more_specific_if',
    'overload_method_deleted_args',
    'overload_method_more_args',
    'same_function_less_args',
    'same_function_more_args',
    'same_function_swap_args',
    'same_function_wrong_caller',
    'swap_arguments',
    'swap_boolean_literal',
    'wrong_function_name'
     'Null Pointer Dereference'
     'Resource Leak'
     'Thread Safety Violation'
]