# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

import sys

sys.path.append('..')
from text2vec import SentenceModel, cos_sim, semantic_search, Similarity, EncoderType

if __name__ == '__main__':
    m = SentenceModel("shibing624/text2vec-base-multilingual")
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
        'A cheetah is running behind its prey.'
        'The quick brown fox jumps over the lazy dog.',
    ]
    # 1. Compute text embedding
    corpus_embeddings = m.encode(corpus)
    print(type(corpus_embeddings), corpus_embeddings.shape)
    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(corpus, corpus_embeddings):
        print("Sentence:", sentence)
        print("Embedding shape:", embedding.shape)
        print("Embedding head:", embedding[:10])
        print()

    # 2. Compute cosine-similarities for sentence1 and sentence2
    sim_model = Similarity("shibing624/text2vec-base-multilingual")
    cosine_scores = sim_model.get_score(corpus[0], corpus[1])
    print('{} vs {} cos score: {:.4f}'.format(corpus[0], corpus[1], cosine_scores))
    # 以上相似度计算的实质是对embedding结果求cos值，等同于：
    cosine_scores = cos_sim(corpus_embeddings[0], corpus_embeddings[1])
    print('{} vs {} cos score: {:.4f}'.format(corpus[0], corpus[1], float(cosine_scores[0])))

    print('#' * 42)
    # 3. Use semantic_search to perform cosine similarty + topk
    # Query sentences:
    queries = [
        '如何更换花呗绑定银行卡',
        'A man is eating pasta.',
        'Someone in a gorilla costume is playing a set of drums.',
        'A cheetah chases prey on across a field.',
        '敏捷的棕色狐狸跳过了懒狗',
    ]
    for query in queries:
        query_embedding = m.encode(query)
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
        hits = hits[0]  # Get the hits for the first query
        for hit in hits:
            print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
