import json
import random
import os


def shuffle_and_split(input_file, output_dir, ratios=(1, 4, 5)):
    """
    打乱数据集顺序并按给定比例进行划分，并以 JSON 格式保存。

    :param input_file: 输入文件路径（JSONL格式）
    :param output_dir: 输出目录，划分后的数据集将保存到该目录
    :param ratios: 划分比例，默认1:4:5
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = [json.loads(line.strip()) for line in infile]

    # 打乱数据集顺序
    random.shuffle(data)

    # 计算每一部分的大小
    total_size = len(data)
    ratio_sum = sum(ratios)
    split_sizes = [int(ratio / ratio_sum * total_size) for ratio in ratios]

    # 确保最后一部分包含剩余的数据
    split_sizes[-1] += total_size - sum(split_sizes)

    # 按照比例划分数据
    train_data = data[:split_sizes[0]]
    val_data = data[split_sizes[0]:split_sizes[0] + split_sizes[1]]
    test_data = data[split_sizes[0] + split_sizes[1]:]

    # 创建输出目录（如果不存在的话）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存划分后的数据集为 JSON 格式
    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, 'database.json'), 'w', encoding='utf-8') as val_file:
        json.dump(val_data, val_file, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, 'contra_learn.json'), 'w', encoding='utf-8') as test_file:
        json.dump(test_data, test_file, ensure_ascii=False, indent=4)

    print(f"Data split completed. test: {len(train_data)}, database: {len(val_data)}, contra_learn: {len(test_data)}")


if __name__ == '__main__':
    # 示例使用
    input_file = '/autodl-fs/data/dataSet/pyData/likely_bug_true.jsonl'
    output_dir = '/autodl-fs/data/dataSet/pyData/split_data'
    shuffle_and_split(input_file, output_dir)

