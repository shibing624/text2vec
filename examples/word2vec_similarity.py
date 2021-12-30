# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from text2vec import Similarity, cos_sim

sim = Similarity()

# Two lists of sentences
sentences1 = ['如何更换花呗绑定银行卡',
              'The cat sits outside',
              'A man is playing guitar',
              'The new movie is awesome']

sentences2 = ['花呗更改绑定银行卡',
              'The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

# Output the pairs with their score
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        score = sim.get_score(sentences1[i], sentences2[j])
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], score))
print()

sim2 = Similarity(similarity_type='wmd', embedding_type='w2v')
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        score = sim2.get_score(sentences1[i], sentences2[j])
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], score))
print()

sim3 = Similarity(similarity_type='cosine', embedding_type='sbert')
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        score = sim3.get_score(sentences1[i], sentences2[j])
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], score))
print()
