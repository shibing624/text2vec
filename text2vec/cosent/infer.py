# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import numpy as np
from loguru import logger
from transformers import BertTokenizer
sys.path.append('../..')
from text2vec.cosent.data_helper import load_test_data
from text2vec.cosent.model import Model
from text2vec.cosent.train import set_args, compute_corrcoef, evaluate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def sbert_cos(model_dir, data_path):
    from sentence_transformers import SentenceTransformer, util
    sents1, sents2, labels = load_test_data(data_path)
    m = SentenceTransformer(model_dir)
    e1 = m.encode(sents1)
    e2 = m.encode(sents2)
    s = util.cos_sim(e1, e2)
    sims = []
    for i in range(len(sents1)):
        sims.append(s[i][i])
    sims = np.array(sims)
    labels = np.array(labels)
    corrcoef = compute_corrcoef(labels, sims)
    logger.debug(f'labels: {labels[:10]}')
    logger.debug(f'sims: {sims[:10]}')
    logger.debug(f'Spearman corr: {corrcoef}')
    return corrcoef


if __name__ == '__main__':
    args = set_args()
    logger.info(args)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir)
    model = Model(args.output_dir)
    corr = evaluate(model, tokenizer, args.test_path)
    print(corr)

    corr = sbert_cos(args.output_dir, args.test_path)
    print(corr)
