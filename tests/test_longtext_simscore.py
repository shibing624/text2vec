# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')
from text2vec import SentenceModel, cos_sim

sbert_model = SentenceModel()

a = '你们都喜欢火影忍者里的谁啊，你说的到底是谁？看Bert里面extract_features.py这个文件，可以得到类似预训练的词向量组成的句子表示，' \
    '类似于Keras里面第一步Embedding层。以题主所说的句子相似度计算为例，只需要把两个句子用分隔符隔开送到bert的输入（首位加特殊标记符' \
    'CLS的embedding），然后取bert输出中和CLS对应的那个vector（记为c）进行变换就可以了。原文中提到的是多分类任务，给出的输出变换是' \
    '）就可以了。至于题主提到的句向量表示，上文中提到的向量c即可一定程度表' \
    '示整个句子的语义，原文中有提到“ The final hidden state (i.e., output of Transformer) corresponding to this token ' \
    'is used as the aggregate sequence representation for classification tasks.”' \
    '这句话中的“this token”就是CLS位。补充：除了直接使用bert的句对匹配之外，还可以只用bert来对每个句子求embedding。之后再通过向' \
    'Siamese Network这样的经典模式去求相似度也可以'

b = '你说的到底是谁？看Bert里面extract_features.py这个文件，可以得到类似预训练的词向量组成的句子表示，' \
    '类似于Keras里面第一步Embedding层。以题主所说的句子相似度计算为例，只需要把两个句子用分隔符隔开送到bert的输入（首位加特殊标记符' \
    'CLS的embedding），然后取bert输出中和CLS对应的那个vector（记为c）进行变换就可以了。原文中提到的是多分类任务，给出的输出变换是' \
    '）就可以了。至于题主提到的句向量表示，上文中提到的向量c即可一定程度表'


def sbert_sim_score(str_a, str_b):
    a_emb = sbert_model.encode(str_a)
    b_emb = sbert_model.encode(str_b)
    return cos_sim(a_emb, b_emb).item()


class TestCase(unittest.TestCase):
    def test_bert_sim(self):
        r = sbert_sim_score(a, b)
        print(r)
        self.assertEqual(abs(r - 0.872) < 0.2, True)


if __name__ == '__main__':
    unittest.main()
