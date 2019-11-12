# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from simtext.embeddings.word_embedding import WordEmbedding

if __name__ == '__main__':
    b = WordEmbedding(w2v_path='~/Codes/ai-server-xuming/data/sentence_w2v.bin', w2v_kwargs={'binary': True})
    data1 = '你 好 啊'.split(' ')
    r = b.embed([data1], True)

    print(r)
    print(r.shape)
