# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from simtext.embeddings.word_embedding import WordEmbedding

if __name__ == '__main__':
    b = WordEmbedding()
    data1 = '你 好 啊'.split(' ')
    r = b.embed([data1], True)

    print(r)
    print(r.shape)
