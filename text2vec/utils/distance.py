# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from difflib import SequenceMatcher

import numpy as np

zero_bit = 0.000000001


def try_divide(x, y, val=0.0):
    """
    try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val


def cosine_distance(v1, v2):
    """
    余弦距离
    return cos score
    """
    up = float(np.sum(v1 * v2))
    down = np.linalg.norm(v1) * np.linalg.norm(v2)
    return try_divide(up, down)

def hamming_distance(v1, v2):
    n = int(v1, 2) ^ int(v2, 2)
    return bin(n & 0xffffffff).count('1')


def euclidean_distance(v1, v2):  # 欧氏距离
    return np.sqrt(np.sum(np.square(v1 - v2)))


def manhattan_distance(v1, v2):  # 曼哈顿距离
    return np.sum(np.abs(v1 - v2))


def chebyshev_distance(v1, v2):  # 切比雪夫距离
    return np.max(np.abs(v1 - v2))


def minkowski_distance(v1, v2):  # 闵可夫斯基距离
    return np.sqrt(np.sum(np.square(v1 - v2)))


def euclidean_distance_standardized(v1, v2):  # 标准化欧氏距离
    v1_v2 = np.vstack([v1, v2])
    sk_v1_v2 = np.var(v1_v2, axis=0, ddof=1)
    return np.sqrt(((v1 - v2) ** 2 / (sk_v1_v2 + zero_bit * np.ones_like(sk_v1_v2))).sum())


def edit_distance(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        import Levenshtein
        d = Levenshtein.distance(str1, str2) / float(max(len(str1), len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x == " ", str1, str2).ratio()
    return d


def pearson_correlation_distance(v1, v2):  # 皮尔逊相关系数（Pearson correlation）
    v1_v2 = np.vstack([v1, v2])
    return np.corrcoef(v1_v2)[0][1]


def jaccard_similarity_coefficient_distance(v1, v2):  # 杰卡德相似系数(Jaccard similarity coefficient)
    # 公式求解
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    up = np.double(np.bitwise_and((v1 != v2), np.bitwise_or(v1 != 0, v2 != 0)).sum())
    down = np.double(np.bitwise_or(v1 != 0, v2 != 0).sum() + zero_bit)
    return try_divide(up, down)


def wmd_distance(model, sent1_cut_list, sent2_cut_list):  # WMD距离
    """
    wmd 距离
    :param model: gensim word2vec model
    :param sent1_cut_list:
    :param sent2_cut_list:
    :return:
    """
    distance = model.wmdistance(sent1_cut_list, sent2_cut_list)
    return distance


def is_str_match(str1, str2, threshold=1.0):
    assert 0.0 <= threshold <= 1.0, "Wrong threshold."
    if float(threshold) == 1.0:
        return str1 == str2
    else:
        return (1. - edit_distance(str1, str2)) >= threshold


def longest_match_size(str1, str2):
    sq = SequenceMatcher(lambda x: x == " ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def longest_match_ratio(str1, str2):
    sq = SequenceMatcher(lambda x: x == " ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return try_divide(match.size, min(len(str1), len(str2)))


def jaccard_coef(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return try_divide(float(len(A.intersection(B))), len(A.union(B)))


def num_of_common_sub_str(str1, str2):
    """
    求两个字符串的最长公共子串
    思想：建立一个二维数组，保存连续位相同与否的状态
    """
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    max_num = 0  # 最长匹配长度

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > max_num:
                    # 获取最大匹配长度
                    max_num = record[i + 1][j + 1]
    return max_num


def string_hash(source):
    if source == "":
        return 0
    else:
        x = ord(source[0]) << 7
        m = 1000003
        mask = 2 ** 128 - 1
        for c in source:
            x = ((x * m) ^ ord(c)) & mask
        x ^= len(source)
        if x == -1:
            x = -2
        x = bin(x).replace('0b', '').zfill(64)[-64:]

        return str(x)


def sim_hash(text):
    import jieba.analyse
    seg = jieba.cut(text)
    key_word = jieba.analyse.extract_tags('|'.join(seg), topK=20, withWeight=True, allowPOS=())
    # 先按照权重排序，再按照词排序
    key_list = []
    for feature, weight in key_word:
        weight = int(weight * 20)
        temp = []
        for f in string_hash(feature):
            if f == '1':
                temp.append(weight)
            else:
                temp.append(-weight)
        key_list.append(temp)
    content_list = np.sum(np.array(key_list), axis=0)
    # 编码读不出来
    if len(key_list) == 0:
        return '00'
    simhash = ''
    for c in content_list:
        if c > 0:
            simhash = simhash + '1'
        else:
            simhash = simhash + '0'
    return simhash


def normalization(x):
    """
      归一化，最大最小值
    :param x:
    :return:
    """
    return [(float(i) - min(x)) / float(max(x) - min(x) + zero_bit) for i in x]


def z_score(x, axis=0):
    """
      标准化
    :param x: arrary, numpy
    :param axis: int, 0
    :return: arrary, numpy
    """
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    return x



if __name__ == '__main__':
    vec1_test = np.array([1, 38, 17, 32])
    vec2_test = np.array([5, 6, 8, 9])

    str1_test = "你到底是谁?"
    str2_test = "没想到我是谁，是真样子"

    print(euclidean_distance(vec1_test, vec2_test))
    print(cosine_distance(vec1_test, vec2_test))
    print(manhattan_distance(vec1_test, vec2_test))
    print(euclidean_distance(vec1_test, vec2_test))
    print(cosine_distance(vec1_test, vec2_test))

    print('hamming_distance:', str1_test, str2_test, hamming_distance(sim_hash(str1_test), sim_hash(str2_test)))
    print(edit_distance(str1_test, str2_test))
    print(num_of_common_sub_str(str1_test, str2_test))
    print(normalization(vec1_test))  # 归一化（0-1）
    print(z_score(vec1_test))  # 标准化（0附近，正负）
