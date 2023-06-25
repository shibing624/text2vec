# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 文本语义相似度计算
"""
import sys

sys.path.append('..')
from text2vec import SentenceModel, cos_sim, Similarity, SimilarityType, EmbeddingType

# Two lists of sentences
sentences1 = [
    '如何更换花呗绑定银行卡',
    'The cat sits outside',
    'A man is playing guitar',
    'The new movie is awesome',
    '敏捷的棕色狐狸跳过了懒狗',
]

sentences2 = [
    '花呗更改绑定银行卡',
    'The dog plays in the garden',
    'A woman watches TV',
    'The new movie is so great',
    'The quick brown fox jumps over the lazy dog.',
]

sim_model = Similarity("shibing624/text2vec-base-multilingual")
# 纯中文建议使用模型"shibing624/text2vec-base-chinese"，
# 多语言建议使用模型"shibing624/text2vec-base-multilingual"
scores = sim_model.get_scores(sentences1, sentences2)
print('1:use Similarity compute cos scores\n')
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], scores[i][j]))
print()

print('-' * 42)
# 以上相似度计算的逻辑，分解为：
# Load pre-trained Sentence Transformer Model (based on DistilBERT). It will be downloaded automatically
model = SentenceModel()

# Compute embedding for both lists
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# Compute cosine-similarits
cosine_scores = cos_sim(embeddings1, embeddings2)
print('2:same to 1, use cos_sim compute cos scores\n')
# Output the pairs with their score
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], cosine_scores[i][j]))
print()

print('-' * 42)
print('3:use Word2Vec compute wmd similarity scores\n')
# 使用Word2Vec计算wmd相似度
sim2 = Similarity("w2v-light-tencent-chinese",
                  similarity_type=SimilarityType.WMD,
                  embedding_type=EmbeddingType.WORD2VEC)
scores = sim2.get_scores(sentences1, sentences2)
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], scores[i][j]))
print()

print('-' * 42)
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
print('4:find out the pairs with the highest cosine similarity scores\n')
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
