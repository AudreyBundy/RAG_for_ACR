


def calculate_precision(true_defect_code, generated_code):
    """
    计算代码片段的查准率（Precision），基于字符串匹配

    :param true_defect_code: 真实缺陷的代码片段
    :param generated_code: 生成的代码片段

    :return: 查准率值
    """
    # 判断生成的代码是否包含真实缺陷代码
    print(true_defect_code)
    print(generated_code)
    true_positive = 0
    false_positive = 0

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

