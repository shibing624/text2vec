# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import time
import numpy as np
from loguru import logger
from transformers import BertTokenizer

sys.path.append('../..')
from text2vec.cosent.data_helper import load_test_data
from text2vec.cosent.model import Model
from text2vec.cosent.train import set_args, compute_corrcoef, evaluate
from text2vec.sbert import SBert, cos_sim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def sbert_cos(model_dir, sents1, sents2, labels):
    m = SBert(model_dir)
    t1 = time.time()
    e1 = m.encode(sents1)
    e2 = m.encode(sents2)
    s = cos_sim(e1, e2)
    sims = []
    for i in range(len(sents1)):
        sims.append(s[i][i])
    sims = np.array(sims)
    spend_time = time.time() - t1
    labels = np.array(labels)
    corrcoef = compute_corrcoef(labels, sims)
    logger.debug(f'labels: {labels[:10]}')
    logger.debug(f'sims: {sims[:10]}')
    logger.debug(f'Spearman corr: {corrcoef}')
    logger.debug(f'spend time: {spend_time}, count:{len(sents1 + sents2)}, qps: {len(sents1 + sents2) / spend_time}')
    return corrcoef


if __name__ == '__main__':
    args = set_args()
    logger.info(args)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir)
    model = Model(args.output_dir)
    t1 = time.time()
    corr = evaluate(model, tokenizer, args.test_path)
    print(corr)
    spend_time = time.time() - t1
    sents1, sents2, labels = load_test_data(args.test_path)
    logger.debug(f'spend time: {spend_time}, count:{len(sents1 + sents2)}, qps: {len(sents1 + sents2) / spend_time}')

    corr = sbert_cos(args.output_dir, sents1, sents2, labels)
    print(corr)
