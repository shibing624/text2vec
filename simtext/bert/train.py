# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import tensorflow as tf

from simtext.bert.model import BertSimilarity

if __name__ == '__main__':
    sim = BertSimilarity(data_dir='../data/', model_dir='/Users/xuming06/.simtext/datasets/chinese_L-12_H-768_A-12', output_dir='../output')
    sim.set_mode(tf.estimator.ModeKeys.TRAIN)
    sim.train()
    sim.set_mode(tf.estimator.ModeKeys.EVAL)
    sim.eval()
