# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""
from text2vec.algorithm.rank_bm25 import BM25Okapi
from text2vec.utils.tokenizer import segment

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

query = "windy London"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

a = bm25.get_top_n(tokenized_query, corpus, n=2)
print(a)

print("*" * 45)

corpus = ['女网红能火的只是一小部分', '当下最火的男明星为鹿晗', "How is the weather today?"]
tokenized_corpus = [segment(doc) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

query = '当下最火的女网红是谁？'
tokenized_query = segment(query)

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

a = bm25.get_top_n(tokenized_query, corpus, n=2)
print(a)
