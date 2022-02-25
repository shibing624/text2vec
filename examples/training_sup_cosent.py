# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys
import time
import numpy as np
from loguru import logger

sys.path.append('..')

from text2vec.cosent.cosent_model import CosentModel
from text2vec.cosent.cosent_dataset import load_test_data
from text2vec import cos_sim, compute_spearmanr, EncoderType


def calc_similarity_scores(args, sents1, sents2, labels):
    m = CosentModel(args.output_dir, encoder_type=args.encoder_type, max_seq_length=args.max_seq_length)
    t1 = time.time()
    e1 = m.encode(sents1)
    e2 = m.encode(sents2)
    spend_time = time.time() - t1
    s = cos_sim(e1, e2)
    sims = []
    for i in range(len(sents1)):
        sims.append(s[i][i])
    sims = np.array(sims)
    labels = np.array(labels)
    spearman = compute_spearmanr(labels, sims)
    logger.debug(f'labels: {labels[:10]}')
    logger.debug(f'preds:  {sims[:10]}')
    logger.debug(f'Spearman: {spearman}')
    logger.debug(
        f'spend time: {spend_time:.4f}, count:{len(sents1 + sents2)}, qps: {len(sents1 + sents2) / spend_time}')
    return spearman


def main():
    parser = argparse.ArgumentParser('CoSENT Text Matching task')
    parser.add_argument('--model_name', default='hfl/chinese-macbert-base', type=str,
                        help='Model Arch, such as: hfl/chinese-macbert-base')
    parser.add_argument('--train_file', default='data/STS-B/STS-B.train.data', type=str, help='Train data path')
    parser.add_argument('--valid_file', default='data/STS-B/STS-B.valid.data', type=str, help='Valid data path')
    parser.add_argument('--test_file', default='data/STS-B/STS-B.test.data', type=str, help='Test data path')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument('--output_dir', default='./outputs/STS-B-cosent', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=64, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--encoder_type', default='FIRST_LAST_AVG', type=lambda t: EncoderType[t],
                        choices=list(EncoderType), help='Encoder type, string name of EncoderType')
    args = parser.parse_args()
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        model = CosentModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                            max_seq_length=args.max_seq_length)
        model.train_model(args.train_file,
                          args.output_dir,
                          eval_file=args.valid_file,
                          num_epochs=args.num_epochs,
                          batch_size=args.batch_size,
                          lr=args.learning_rate)
        logger.info(f"Model saved to {args.output_dir}")
    if args.do_predict:
        model = CosentModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                            max_seq_length=args.max_seq_length)
        test_data = load_test_data(args.test_file)
        test_data = test_data[:100]

        # Predict embeddings
        srcs = []
        trgs = []
        labels = []
        for terms in test_data:
            src, trg, label = terms[0], terms[1], terms[2]
            srcs.append(src)
            trgs.append(trg)
            labels.append(label)
        logger.debug(f'{test_data[0]}')
        sentence_embeddings = model.encode(srcs)
        logger.debug(f"{type(sentence_embeddings)}, {sentence_embeddings.shape}, {sentence_embeddings[0].shape}")
        # Predict similarity scores
        calc_similarity_scores(args, srcs, trgs, labels)


if __name__ == '__main__':
    main()
