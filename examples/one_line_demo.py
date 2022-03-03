# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Basic example of using SentenceModel
"""

import sys

sys.path.append('..')

import shutil
from text2vec import SentenceModel
from text2vec import CosentModel


def sample_train_demo():
    m = CosentModel("bert-base-chinese")
    print(m)
    temp_dir = "./temp"
    m.train_model(use_hf_dataset=True, num_epochs=1, output_dir=temp_dir)
    r = m.encode(["我爱北京天安门"])
    print(r)

    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    m = SentenceModel()
    corpus_embeddings = m.encode('花呗更改绑定银行卡')
    print(type(corpus_embeddings), corpus_embeddings.shape)
    print(corpus_embeddings)

    sample_train_demo()
