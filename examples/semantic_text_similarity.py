# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from text2vec import SBert
from sentence_transformers.util import cos_sim

# Load pre-trained Sentence Transformer Model (based on DistilBERT). It will be downloaded automatically
model = SBert('paraphrase-multilingual-MiniLM-L12-v2')

# Two lists of sentences
sentences1 = ['如何更换花呗绑定银行卡',
              'The cat sits outside',
              'A man is playing guitar',
              'The new movie is awesome']

sentences2 = ['花呗更改绑定银行卡',
              'The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

# Compute embedding for both lists
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# Compute cosine-similarits
cosine_scores = cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
print()

########### find out the pairs with the highest cosine similarity scores:
# Single list of sentences
sentences = [
    '如何更换花呗绑定银行卡',
    '花呗更改绑定银行卡',
    '我在北京打篮球',
    '我是北京人，我喜欢篮球',
    'The cat sits outside',
    'A man is playing guitar',
    'I love pasta',
    'The new movie is awesome',
    'The cat plays in the garden',
    'A woman watches TV',
    'The new movie is so great',
    'Do you like pizza?']

# Compute embeddings
embeddings = model.encode(sentences)

# Compute cosine-similarities for each sentence with each other sentence
cosine_scores = cos_sim(embeddings, embeddings)

# Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores) - 1):
    for j in range(i + 1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

# Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:10]:
    i, j = pair['index']
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
