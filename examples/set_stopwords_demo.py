# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""

import text2vec
from text2vec import Vector

text2vec.set_log_level('DEBUG')
if __name__ == '__main__':
    sent1 = '我去银行开卡'
    sent2 = '到银行开卡了。'

    vec = Vector()
    emb1 = vec.encode(sent1)
    emb2 = vec.encode(sent2)
    print('check is same:', emb1 == emb2)
    assert emb1 is not emb2

    # load custom stopwords
    custom_stopwords_file = './my_stopwords.txt'
    new_vec = Vector()
    new_vec.set_stopwords_file(custom_stopwords_file)
    emb1 = new_vec.encode(sent1)
    emb2 = new_vec.encode(sent2)
    print('check is same:', emb1 == emb2)
