# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys
import time

import numpy as np
from datasets import load_dataset
from loguru import logger

sys.path.append('..')

from text2vec import CosentModel, SentenceBertModel, BertMatchModel
from text2vec import cos_sim, compute_spearmanr, EncoderType


def calc_similarity_scores(model, sents1, sents2, labels):
    t1 = time.time()
    e1 = model.encode(sents1)
    e2 = model.encode(sents2)
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
    parser = argparse.ArgumentParser('Text Matching task')
    parser.add_argument('--model_arch', default='cosent', const='cosent', nargs='?',
                        choices=['cosent', 'sentencebert', 'bert'], help='model architecture')
    parser.add_argument('--task_name', default='STS-B', const='STS-B', nargs='?',
                        choices=['ATEC', 'STS-B', 'BQ', 'LCQMC', 'PAWSX'], help='task name of dataset')
    parser.add_argument('--model_name', default='hfl/chinese-macbert-base', type=str,
                        help='Transformers model model or path')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument('--output_dir', default='./outputs/STS-B-model', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--encoder_type', default='MEAN', type=lambda t: EncoderType[t],
                        choices=list(EncoderType), help='Encoder type, string name of EncoderType')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        if args.model_arch == 'cosent':
            model = CosentModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                                max_seq_length=args.max_seq_length)
        elif args.model_arch == 'sentencebert':
            model = SentenceBertModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                                      max_seq_length=args.max_seq_length)
        else:
            model = BertMatchModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                                   max_seq_length=args.max_seq_length)

        model.train_model(
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            use_hf_dataset=True,
            hf_dataset_name=args.task_name,
        )
        logger.info(f"Model saved to {args.output_dir}")
    if args.do_predict:
        if args.model_arch == 'cosent':
            model = CosentModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                                max_seq_length=args.max_seq_length)
        elif args.model_arch == 'sentencebert':
            model = SentenceBertModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                                      max_seq_length=args.max_seq_length)
        else:
            model = BertMatchModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                                   max_seq_length=args.max_seq_length)
        test_dataset = load_dataset("shibing624/nli_zh", args.task_name, split="test")
        # Predict embeddings
        srcs = []
        trgs = []
        labels = []
        for terms in test_dataset:
            src, trg, label = terms['sentence1'], terms['sentence2'], terms['label']
            srcs.append(src)
            trgs.append(trg)
            labels.append(label)
        logger.debug(f'{test_dataset[0]}')
        sentence_embeddings = model.encode(srcs)
        logger.debug(f"{type(sentence_embeddings)}, {sentence_embeddings.shape}, {sentence_embeddings[0].shape}")
        # Predict similarity scores
        calc_similarity_scores(model, srcs, trgs, labels)


if __name__ == '__main__':
    main()
