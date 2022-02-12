# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from text2vec import Word2Vec

if __name__ == '__main__':
    sent1 = '我去银行开卡'
    sent2 = '到银行开卡了。'

    vec = Word2Vec()
    emb1 = vec.encode(sent1)
    emb2 = vec.encode(sent2)
    print('check is same:', emb1 == emb2)
    assert emb1 is not emb2

    # load custom stopwords
    custom_stopwords = ['我', '去', '到', '了', '。']
    new_vec = Word2Vec(stopwords=custom_stopwords)
    emb1 = new_vec.encode(sent1)
    emb2 = new_vec.encode(sent2)
    print('check is same:', emb1 == emb2)
