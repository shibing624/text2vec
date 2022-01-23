# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
from time import time

sys.path.append('..')
from text2vec import Word2Vec, SBert
from text2vec.cosent.data_helper import load_test_data

pwd_path = os.path.abspath(os.path.dirname(__file__))
sts_test_path = os.path.join(pwd_path, '../text2vec/data/STS-B/STS-B.test.data')


class QPSEncoderTestCase(unittest.TestCase):
    def test_cosent_speed(self):
        """测试cosent_speed"""
        sents1, sents2, labels = load_test_data(sts_test_path)
        m = SBert('shibing624/text2vec-base-chinese')
        sents = sents1 + sents2
        print('sente size:', len(sents))
        t1 = time()
        m.encode(sents)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('cosent_sbert qps:', len(sents) / spend_time)

    def test_cosent_origin_speed(self):
        """测试cosent_origin_speed"""
        from text2vec.cosent.train import evaluate, get_sent_id_tensor
        from text2vec.cosent.model import Model
        import torch
        from tqdm import tqdm
        from transformers import BertTokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sents1, sents2, labels = load_test_data(sts_test_path)
        sents = sents1 + sents2
        print('sente size:', len(sents))
        model_name = 'shibing624/text2vec-base-chinese'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = Model(model_name)
        model.to(device)
        model.eval()
        t1 = time()
        for s1, s2, lab in tqdm(zip(sents1, sents2, labels)):
            input_ids, input_mask, token_type_ids = get_sent_id_tensor(tokenizer, [s1, s2], 64)
            with torch.no_grad():
                output = model(input_ids, input_mask, token_type_ids)
        t2 = time()
        spend_time = t2 - t1
        print('spend time:', spend_time, ' seconds')
        print('cosent_single qps:', len(sents) / spend_time)
        input_ids, input_mask, token_type_ids = get_sent_id_tensor(tokenizer, sents, 64)
        with torch.no_grad():
            output = model(input_ids, input_mask, token_type_ids)
        t3 = time()
        spend_time = t3 - t2
        print('spend time:', spend_time, ' seconds')
        print('cosent_batch qps:', len(sents) / spend_time)

    def test_sbert_speed(self):
        """测试sbert_speed"""
        sents1, sents2, labels = load_test_data(sts_test_path)
        m = SBert()
        sents = sents1 + sents2
        print('sente size:', len(sents))
        t1 = time()
        m.encode(sents)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('sbert qps:', len(sents) / spend_time)

    def test_w2v_speed(self):
        """测试w2v_speed"""
        sents1, sents2, labels = load_test_data(sts_test_path)
        m = Word2Vec()
        sents = sents1 + sents2
        print('sente size:', len(sents))
        t1 = time()
        m.encode(sents)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('w2v qps:', len(sents) / spend_time)


if __name__ == '__main__':
    unittest.main()
