# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')
from text2vec import SentenceModel


def use_transformers(sentences=('如何更换花呗绑定银行卡', '花呗更改绑定银行卡')):
    from transformers import BertTokenizer, BertModel
    import torch

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Load model from HuggingFace Hub
    tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
    model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print(sentence_embeddings.shape)
    return sentence_embeddings


class SBERTEmbeddingsTestCase(unittest.TestCase):
    def test_encode_text(self):
        """测试文本 text encode结果"""
        a = '如何更换花呗绑定银行卡'
        m = SentenceModel('shibing624/text2vec-base-chinese')
        emb = m.encode(a)
        print(a)
        self.assertEqual(emb.shape, (768,))

    def test_tr_emb(self):
        """测试test_tr_emb"""
        r = use_transformers()
        print(r.shape)
        print("Sentence embeddings:")
        print(r)


    def test_sbert_encode_text(self):
        """测试test_sbert_encode_text"""
        a = '如何更换花呗绑定银行卡'
        m = SentenceModel('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        emb = m.encode(a)
        print(a)
        self.assertEqual(emb.shape, (384,))

    def test_sbert_dim(self):
        m = SentenceModel('shibing624/text2vec-base-chinese')
        print('dim:', m.bert.pooler.dense.out_features)
        def get_sentence_embedding_dimension(model):
            if model:
                sent_embedding_dim_method = getattr(model.bert.pooler.dense, "out_features", None)
                if sent_embedding_dim_method:
                    return sent_embedding_dim_method
            return None
        dim = get_sentence_embedding_dimension(m)
        print(dim)


if __name__ == '__main__':
    unittest.main()
