# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from text2vec.embeddings.word_embedding import WordEmbedding

if __name__ == '__main__':
    import text2vec

    char = '，'
    result = text2vec.encode(char)
    print(type(result))
    print(char, result)


    b = WordEmbedding()
    data1 = '你 好 啊'.split(' ')
    r = b.embed([data1], True)

    print(r)
    print(r.shape)


