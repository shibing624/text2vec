# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 语义相似文本检索
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
import sys

sys.path.append('..')

from text2vec import SentenceModel, cos_sim, semantic_search, BM25
import torch

embedder = SentenceModel("shibing624/text2vec-base-multilingual")

# Corpus with example sentences
corpus = [
    '花呗更改绑定银行卡',
    '我什么时候开通了花呗',
    'A man is eating food.',
    'A man is eating a piece of bread.',
    'The girl is carrying a baby.',
    'A man is riding a horse.',
    'A woman is playing violin.',
    'Two men pushed carts through the woods.',
    'A man is riding a white horse on an enclosed ground.',
    'A monkey is playing drums.',
    'A cheetah is running behind its prey.',
    'The quick brown fox jumps over the lazy dog.',
]
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = [
    '如何更换花呗绑定银行卡',
    'A man is eating pasta.',
    'Someone in a gorilla costume is playing a set of drums.',
    'A cheetah chases prey on across a field.',
    '敏捷的棕色狐狸跳过了懒狗',
]

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
print('\nuse cos sim calc each query and corpus:')
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))

print('#' * 42)
########  use semantic_search to perform cosine similarty + topk
print('\nuse semantic_search to perform cosine similarty + topk:')

for query in queries:
    query_embedding = embedder.encode(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

print('#' * 42)
######## use bm25 to rank search score
print('\nuse bm25 to calc each score:')

search_sim = BM25(corpus=corpus)
for query in queries:
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    for i in search_sim.get_scores(query, top_k=5):
        print(i[0], "(Score: {:.4f})".format(i[1]))
