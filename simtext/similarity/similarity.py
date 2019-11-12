# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import jieba

from simtext.embeddings.bert_embedding import BERTEmbedding
from simtext.embeddings.word_embedding import WordEmbedding
from simtext.utils.distance import cos_dist


class EmbType(object):
    BERT = 'bert'
    W2V = 'w2v'


class Similarity(object):
    def __init__(self, embedding_type='bert'):
        self.embedding_type = embedding_type

    def similarity_score(self, text1, text2):
        ret = 0.0
        if not text1.strip() or not text2.strip():
            return ret

        if self.embedding_type == EmbType.BERT:
            b = BERTEmbedding(model_folder='/Users/xuming06/Codes/bert/data/chinese_L-12_H-768_A-12',
                              sequence_length=128)
            tokens_1 = b.tokenizer.tokenize(text1)
            tokens_2 = b.tokenizer.tokenize(text2)
            emb_1 = b.embed([tokens_1])[0]
            emb_2 = b.embed([tokens_2])[0]
            ret = cos_dist(emb_1, emb_2)
        elif self.embedding_type == EmbType.W2V:
            b = WordEmbedding(w2v_path='/Users/xuming06/Codes/ai-server-xuming/data/sentence_w2v.bin',
                              w2v_kwargs={'binary': True})
            tokens_1 = jieba.lcut(text1)
            tokens_2 = jieba.lcut(text2)
            emb_1 = b.embed([tokens_1])[0]
            emb_2 = b.embed([tokens_2])[0]
            ret = cos_dist(emb_1, emb_2)
        else:
            raise ValueError('set error embedding_type.')
        return ret


if __name__ == '__main__':
    sim = Similarity('bert')
    a = '湖北人爱吃鱼'
    b = '甘肃人不爱吃鱼'
    s = sim.similarity_score(a, b)
    print(s)
