# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 配置切词器
"""
import logging
import os

import jieba
from jieba import posseg

jieba.setLogLevel(log_level="ERROR")


def segment(sentence, cut_type='word', pos=False):
    """
    切词
    :param sentence:
    :param cut_type: 'word' use jieba.lcut; 'char' use list(sentence)
    :param pos: enable POS
    :return: list
    """
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


class JiebaTokenizer(object):
    def __init__(self, dict_path='', custom_word_freq_dict=None):
        self.model = jieba
        self.model.default_logger.setLevel(logging.ERROR)
        # 初始化大词典
        if os.path.exists(dict_path):
            self.model.set_dictionary(dict_path)
        # 加载用户自定义词典
        if custom_word_freq_dict:
            for w, f in custom_word_freq_dict.items():
                self.model.add_word(w, freq=f)

    def tokenize(self, sentence, cut_all=False, HMM=True):
        """
        切词并返回切词位置
        :param sentence: 句子
        :param cut_all: 全模式，默认关闭
        :param HMM: 是否打开NER识别，默认打开
        :return:  A list of strings.
        """
        return self.model.lcut(sentence, cut_all=cut_all, HMM=HMM)
