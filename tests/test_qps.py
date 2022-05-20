# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
from loguru import logger
import time
import os
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.append('..')
from text2vec import Word2Vec, SentenceModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pwd_path = os.path.abspath(os.path.dirname(__file__))
logger.add('test.log')

data = ['如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡']
print("data:", data)
num_tokens = sum([len(i) for i in data])


class TransformersEncoder:
    def __init__(self, model_name='shibing624/text2vec-base-chinese'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def encode(self, sentences):
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings


class QPSEncoderTestCase(unittest.TestCase):
    def test_cosent_speed(self):
        """测试cosent_speed"""
        logger.info("\n---- cosent:")
        model = SentenceModel('shibing624/text2vec-base-chinese')
        for j in range(10):
            tmp = data * (2 ** j)
            c_num_tokens = num_tokens * (2 ** j)
            start_t = time.time()
            r = model.encode(tmp)
            assert r is not None
            print('result shape', r.shape)
            time_t = time.time() - start_t
            logger.info('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
                        (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))

    def test_origin_cosent_speed(self):
        """测试origin_cosent_speed"""
        logger.info("\n---- origin cosent:")
        model = TransformersEncoder('shibing624/text2vec-base-chinese')
        for j in range(10):
            tmp = data * (2 ** j)
            c_num_tokens = num_tokens * (2 ** j)
            start_t = time.time()
            r = model.encode(tmp)
            assert r is not None
            print('result shape', r.shape)
            time_t = time.time() - start_t
            logger.info("----\ncosent:")
            logger.info('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
                        (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))

    def test_sbert_speed(self):
        """测试sbert_speed"""
        logger.info("\n---- sbert:")
        model = SentenceModel('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        for j in range(10):
            tmp = data * (2 ** j)
            c_num_tokens = num_tokens * (2 ** j)
            start_t = time.time()
            r = model.encode(tmp)
            assert r is not None
            print('result shape', r.shape)
            time_t = time.time() - start_t
            logger.info('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
                        (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))

    def test_w2v_speed(self):
        """测试w2v_speed"""
        logger.info("\n---- w2v:")
        model = Word2Vec()
        for j in range(10):
            tmp = data * (2 ** j)
            c_num_tokens = num_tokens * (2 ** j)
            start_t = time.time()
            r = model.encode(tmp)
            assert r is not None
            print('result shape', r.shape)
            time_t = time.time() - start_t
            logger.info('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
                        (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))


if __name__ == '__main__':
    unittest.main()
