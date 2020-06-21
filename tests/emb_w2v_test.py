# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""
import text2vec
from text2vec.embeddings.word_embedding import WordEmbedding


def test_word_emb():
    b = WordEmbedding()
    data1 = '你 好 啊'.split(' ')
    r = b.embed([data1], True)

    print(r)
    print(r.shape)


def test_oov_emb():
    char = '，'
    result = text2vec.encode(char)
    print(char, result)

    char = '特价机票'
    result = text2vec.encode(char)
    print(char, result)

    char = '特价'
    result = text2vec.encode(char)
    print(char, result)

    char = '机票'
    result = text2vec.encode(char)
    print(char, result)


def test_oov_sim():
    from text2vec import Similarity
    sim = Similarity()
    a = '，'
    b = '花'
    s = sim.get_score(a, b)
    print(a, b, s)

    a = '，画画'
    b = '花画画'
    s = sim.get_score(a, b)
    print(a, b, s)

    a = '，'
    b = '花画画'
    s = sim.get_score(a, b)
    print(a, b, s)

    a = '，机票'
    b = '特价机票'
    s = sim.get_score(a, b)
    print(a, b, s)

    a = '机票'
    b = '特价机票'
    s = sim.get_score(a, b)
    print(a, b, s)

    a = '机票'
    b = '特价的机票'
    s = sim.get_score(a, b)
    print(a, b, s)


def test_sentence_emb():
    char = '你'
    result = text2vec.encode(char)
    print(char, result)
    char = '好'
    result = text2vec.encode(char)
    print(char, result)
    char = '吗'
    result = text2vec.encode(char)
    print(char, result)

    char = '你好'
    result = text2vec.encode(char)
    print(char, result)

    char = '你好吗'
    result = text2vec.encode(char)
    print(char, result)

    import numpy as np
    emb = [text2vec.encode('你好'), text2vec.encode('吗')]
    average = np.array(emb).sum(axis=0) / 2.0
    print('average:', average)
    act = text2vec.encode('你好吗')

    if str(act) == str(average):
        print("same")
    else:
        print('diff')
