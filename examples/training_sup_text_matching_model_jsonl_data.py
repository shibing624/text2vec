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

from text2vec import CosentModel, SentenceBertModel, BertMatchModel
from text2vec import cos_sim, compute_spearmanr, EncoderType, load_text_matching_test_data


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
    parser.add_argument('--model_name', default='nghuyong/ernie-3.0-base-zh', type=str,
                        help='Transformers model model or path')
    parser.add_argument('--train_file', default='data/snli_zh_50.jsonl', type=str, help='Train data path')
    parser.add_argument('--valid_file', default='data/snli_zh_50.jsonl', type=str, help='Train data path')
    parser.add_argument('--test_file', default='data/snli_zh_50.jsonl', type=str, help='Test data path')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument('--output_dir', default='./outputs/jsonl-model', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=256, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--save_model_every_epoch', action="store_true", help="Whether to save model after each epoch")
    parser.add_argument('--encoder_type', default='MEAN', type=lambda t: EncoderType[t],
                        choices=list(EncoderType), help='Encoder type, string name of EncoderType')
    parser.add_argument("--bf16", action="store_true", help="Whether to use bfloat16 amp training.")
    parser.add_argument("--data_parallel", action="store_true", help="Whether to use multi-gpu data parallel.")
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        if args.model_arch == 'cosent':
            model = CosentModel(
                model_name_or_path=args.model_name,
                encoder_type=args.encoder_type,
                max_seq_length=args.max_seq_length
            )
        elif args.model_arch == 'sentencebert':
            model = SentenceBertModel(
                model_name_or_path=args.model_name,
                encoder_type=args.encoder_type,
                max_seq_length=args.max_seq_length
            )
        else:
            model = BertMatchModel(
                model_name_or_path=args.model_name,
                encoder_type=args.encoder_type,
                max_seq_length=args.max_seq_length
            )
        model.train_model(
            args.train_file,
            args.output_dir,
            eval_file=args.valid_file,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            save_model_every_epoch=args.save_model_every_epoch,
            bf16=args.bf16,
            data_parallel=args.data_parallel,
        )
        logger.info(f"Model saved to {args.output_dir}")
    if args.do_predict:
        if args.model_arch == 'cosent':
            model = CosentModel(
                model_name_or_path=args.output_dir,
                encoder_type=args.encoder_type,
                max_seq_length=args.max_seq_length
            )
        elif args.model_arch == 'sentencebert':
            model = SentenceBertModel(
                model_name_or_path=args.output_dir,
                encoder_type=args.encoder_type,
                max_seq_length=args.max_seq_length
            )
        else:
            model = BertMatchModel(
                model_name_or_path=args.output_dir,
                encoder_type=args.encoder_type,
                max_seq_length=args.max_seq_length
            )
        test_data = load_text_matching_test_data(args.test_file)

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
        calc_similarity_scores(model, srcs, trgs, labels)


if __name__ == '__main__':
    main()
