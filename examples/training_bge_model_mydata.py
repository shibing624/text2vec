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

from text2vec import BgeModel, SentenceModel
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
    parser.add_argument('--model_arch', default='bge', const='bge', nargs='?',
                        choices=['bge'], help='model architecture')
    parser.add_argument('--model_name', default='BAAI/bge-large-zh-noinstruct', type=str,
                        help='Transformers model model or path')
    parser.add_argument('--train_file', default='data/bge_finetune_data.jsonl', type=str, help='Train data path')
    parser.add_argument('--valid_file', default='data/snli_zh_50.jsonl', type=str, help='Train data path')
    parser.add_argument('--test_file', default='data/snli_zh_50.jsonl', type=str, help='Test data path')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument('--output_dir', default='./outputs/bge-model', type=str, help='Model output directory')
    parser.add_argument('--query_max_len', default=32, type=int, help='Max sequence length for query')
    parser.add_argument('--passage_max_len', default=64, type=int, help='Max sequence length for passage')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--train_group_size', default=4, type=int, help='Train group size')
    parser.add_argument('--temperature', default=1.0, type=float, help='Temperature for softmax')
    parser.add_argument('--save_model_every_epoch', action="store_true", help="Whether to save model after each epoch")
    parser.add_argument('--encoder_type', default='MEAN', type=lambda t: EncoderType[t],
                        choices=list(EncoderType), help='Encoder type, string name of EncoderType')
    parser.add_argument("--bf16", action="store_true", help="Whether to use bfloat16 amp training.")
    parser.add_argument("--data_parallel", action="store_true", help="Whether to use multi-gpu data parallel.")
    parser.add_argument("--normalize_embeddings", action="store_true",
                        help="Whether to normalize embeddings. set True if temperature < 1.0")
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        model = BgeModel(
            model_name_or_path=args.model_name,
            encoder_type=args.encoder_type,
            max_seq_length=args.query_max_len,
            passage_max_len=args.passage_max_len,
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
            train_group_size=args.train_group_size,
            temperature=args.temperature,
            normalize_embeddings=args.normalize_embeddings,
        )
        logger.info(f"Model saved to {args.output_dir}")
    if args.do_predict:
        model = SentenceModel(
            model_name_or_path=args.output_dir,
            encoder_type=args.encoder_type,
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
