# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

import tensorflow as tf

from simtext.bert.model import BertSimilarity
from simtext.utils.logger import get_logger

logger = get_logger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    sim = BertSimilarity(data_dir='../data/', model_dir='/Users/xuming06/.simtext/datasets/chinese_L-12_H-768_A-12',
                         output_dir='../output')
    sim.set_mode(tf.estimator.ModeKeys.PREDICT)
    while True:
        sentence1 = input('sentence1: ')
        sentence2 = input('sentence2: ')
        predict = sim.predict(sentence1, sentence2)
        print(f'similarity：{predict[0][1]}')
