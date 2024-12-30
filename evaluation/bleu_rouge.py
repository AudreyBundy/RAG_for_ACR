import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu



import re
from collections import Counter
import math


def get_overlap_and_diff(str1, str2):
    overlap_length = 0
    for c1, c2 in zip(str1, str2):
        if c1 == c2:
            overlap_length += 1
        else:
            break
    return str1[:overlap_length], str2[overlap_length:]


def deal_fix_code(my_defect_code,defect_code,my_fix,fix):
    _, same_s1 = get_overlap_and_diff(my_defect_code, defect_code)
    _, same_s2 = get_overlap_and_diff(my_fix, fix)

    if same_s1 == same_s2:
        fix = fix.replace(same_s1, "")

    return fix






def tokenize_code(code):
    """简单的代码分词器，将代码按空格、运算符、括号等进行分割"""
    # 用正则表达式将代码中的标识符、关键字、运算符分离出来
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return tokens


def ngram_precision(reference, candidate, n):
    """计算n-gram精确度"""
    ref_ngrams = Counter([tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)])
    cand_ngrams = Counter([tuple(candidate[i:i + n]) for i in range(len(candidate) - n + 1)])

    # 计算候选文本中n-gram的重叠部分
    overlap = sum(min(ref_ngrams[ngram], cand_ngrams[ngram]) for ngram in cand_ngrams)
    total = sum(cand_ngrams.values())

    return overlap / total if total > 0 else 0.0

def brevity_penalty(reference, candidate):
    """计算BLEU中的简短惩罚（Brevity Penalty）"""
    c = len(candidate)  # 生成文本长度
    r = len(reference)  # 参考文本长度
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c) if c > 0 else 0


def calculate_bleu(reference, candidate, n_gram=4):
    """计算BLEU分数，支持多个n-gram，默认为1-gram到4-gram"""
    bleu = 0
    total_precision = 0

    # 计算1-gram到n-gram的精确度
    for n in range(1, n_gram + 1):
        precision = ngram_precision(reference, candidate, n)
        total_precision += math.log(precision) if precision > 0 else 0

    # 计算简短惩罚（Brevity Penalty）
    bp = brevity_penalty(reference, candidate)

    # 计算BLEU分数
    bleu = bp * math.exp(total_precision / n_gram)

    return bleu


# 计算ROUGE分数
# ROUGE 主要通过比较自动生成的文本与参考文本（通常是人工标注的文本）之间的重叠情况来评估文本的质量。
# ROUGE-1：基于 unigram（单词级别的）重叠
# ROUGE-2：基于 bigram（双词组）重叠
# ROUGE-L：基于最长公共子序列（Longest Common Subsequence）的重叠
def calculate_rouge(reference_text, generated_text):
    # 使用rouge-scorer计算ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 计算ROUGE分数
    scores = scorer.score(reference_text, generated_text)

    # 提取各个ROUGE指标的分数
    rouge1_score = scores['rouge1'].fmeasure
    rouge2_score = scores['rouge2'].fmeasure
    rougeL_score = scores['rougeL'].fmeasure

    return rouge1_score, rouge2_score, rougeL_score