# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Basic example of using SentenceModel
"""

import sys

sys.path.append('..')
from text2vec import SentenceModel

if __name__ == '__main__':
    m = SentenceModel()
    corpus_embeddings = m.encode('花呗更改绑定银行卡')
    print(type(corpus_embeddings), corpus_embeddings.shape)
    print(corpus_embeddings)
