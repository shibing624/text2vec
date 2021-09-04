# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

import sys

sys.path.append('..')
from text2vec import SBert, semantic_search

if __name__ == '__main__':
    sbert_model = SBert('paraphrase-multilingual-MiniLM-L12-v2')
    # Corpus with example sentences
    corpus = [
        '卡',
        '银行卡',
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
        'A cheetah is running behind its prey.'
    ]
    corpus_embeddings = sbert_model.encode(corpus)
    print(type(corpus_embeddings), corpus_embeddings.shape)
    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(corpus, corpus_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")

    ########  use semantic_search to perform cosine similarty + topk
    print('#' * 42)
    # Query sentences:
    queries = [
        '如何更换花呗绑定银行卡',
        'A man is eating pasta.',
        'Someone in a gorilla costume is playing a set of drums.',
        'A cheetah chases prey on across a field.'
    ]
    for query in queries:
        query_embedding = sbert_model.encode(query)
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
        hits = hits[0]  # Get the hits for the first query
        for hit in hits:
            print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
