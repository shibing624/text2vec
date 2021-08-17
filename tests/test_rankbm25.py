# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')
from text2vec.utils.rank_bm25 import BM25Okapi
from text2vec.utils.tokenizer import segment


class RankTestCase(unittest.TestCase):
    def test_en_topn(self):
        """测试en文本bm25 topn"""
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
        self.assertTrue(' '.join(["{:.3f}".format(i) for i in doc_scores]) == "0.000 0.937 0.000")

        a = bm25.get_top_n(tokenized_query, corpus, n=2)
        print(a)
        self.assertEqual(a, ['It is quite windy in London', 'How is the weather today?'])

    def test_zh_topn(self):
        """测试zh文本bm25 topn"""
        corpus = ['女网红能火的只是一小部分', '当下最火的男明星为鹿晗', "How is the weather today?", "你觉得哪个女的明星最红？"]
        tokenized_corpus = [segment(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        query = '当下最火的女的明星是谁？'
        tokenized_query = segment(query)
        doc_scores = bm25.get_scores(tokenized_query)
        print(doc_scores)

        a = bm25.get_top_n(tokenized_query, corpus, n=3)
        print(a)
        self.assertEqual(a, ['你觉得哪个女的明星最红？', '当下最火的男明星为鹿晗', '女网红能火的只是一小部分'])


if __name__ == '__main__':
    unittest.main()
