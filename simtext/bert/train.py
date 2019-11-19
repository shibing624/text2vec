# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

import tensorflow as tf

import simtext
from simtext.bert.model import BertSimilarity

if __name__ == '__main__':
    sim = BertSimilarity(data_dir='../data/', model_dir=os.path.join(simtext.USER_DATA_DIR, 'chinese_L-12_H-768_A-12'),
                         output_dir=os.path.join(simtext.USER_DATA_DIR, 'fine_tuned_bert_similarity'))
    sim.set_mode(tf.estimator.ModeKeys.TRAIN)
    sim.train()
    sim.set_mode(tf.estimator.ModeKeys.EVAL)
    sim.eval()
