def is_subsequence(small, large):
    """
    判断small是否是large的子序列
    :param small: 子序列
    :param large: 大序列
    :return: 如果small是large的子序列，返回True；否则返回False
    """
    it = iter(large)
    return all(char in it for char in small)

def calculate_precision(true_defect_code, generated_code):
    """
    计算代码片段的查准率（Precision），基于字符串匹配
    :param true_defect_code: 真实缺陷的代码片段
    :param generated_code: 生成的代码片段
    :return: 查准率值
    """
    true_positive = 0
    false_positive = 0

    # 判断 true_defect_code 是否是 generated_code 的子序列，或者 generated_code 是否是 true_defect_code 的子序列
    if is_subsequence(true_defect_code, generated_code) or is_subsequence(generated_code, true_defect_code):
        true_positive += 1
    else:
        false_positive += 1

    # 计算查准率
    if true_positive + false_positive == 0:
        return 0.0  # 防止除零错误
    precision = true_positive / (true_positive + false_positive)

    return precision

    if true_defect_code in generated_code:
        true_positive += 1
    else:
        false_positive += 1

    # 计算查准率
    if true_positive + false_positive == 0:
        return 0.0  # 防止除零错误
    precision = true_positive / (true_positive + false_positive)

    return precision


# 示例：真实缺陷代码和生成代码

def calculate_defect_type_precision(true_defect_type, generated_defect_type):
    """
    计算缺陷类型的查准率（Precision），基于缺陷类型匹配

    :param true_defect_type: 真实缺陷类型
    :param generated_defect_type: 生成的缺陷类型

    :return: 缺陷类型查准率
    """
    true_positive = 0
    false_positive = 0

    # 判断生成的缺陷类型是否与真实缺陷类型匹配
    if true_defect_type == generated_defect_type:
        true_positive += 1
    else:
        false_positive += 1

    # 计算查准率
    if true_positive + false_positive == 0:
        return 0.0  # 防止除零错误
    precision = true_positive / (true_positive + false_positive)

    return precision