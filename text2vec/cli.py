# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cli for text2vec
"""
import argparse
import sys

import numpy as np
from loguru import logger

sys.path.append('..')
from text2vec.sentence_model import SentenceModel
from text2vec.word2vec import Word2Vec


def main():
    parser = argparse.ArgumentParser(description='text2vec cli')
    parser.add_argument('--input_file', type=str, help='input file path, text file', required=True)
    parser.add_argument('--output_file', type=str, default='text_embs.npy', help='output file path')
    parser.add_argument('--model_type', type=str, default='sentencemodel', help='model type: sentencemodel, word2vec')
    parser.add_argument('--model_name', type=str, default='shibing624/text2vec-base-chinese', help='model name or path')
    parser.add_argument('--encoder_type', type=str, default='MEAN',
                        help='encoder type: MEAN, CLS, POOLER, FIRST_LAST_AVG, LAST_AVG')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
    parser.add_argument('--device', type=str, default=None, help='device: cpu, cuda')
    parser.add_argument('--show_progress_bar', type=bool, default=True, help='show progress bar')
    parser.add_argument('--normalize_embeddings', type=bool, default=True, help='normalize embeddings')

    args = parser.parse_args()
    logger.debug(args)

    sentences = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip())
    logger.info(f'load sentences success. sentences num: {len(sentences)}')

    if args.model_type == 'sentencemodel':
        model = SentenceModel(
            model_name_or_path=args.model_name,
            encoder_type=args.encoder_type,
            max_seq_length=args.max_seq_length,
            device=args.device,
        )
        logger.info(f'load model success. model: {model}')
        """model.encode
        Returns the embeddings for a batch of sentences.

        :param sentences: str/list, Input sentences
        :param batch_size: int, Batch size
        :param show_progress_bar: bool, Whether to show a progress bar for the sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which device to use for the computation
        :param normalize_embeddings: If true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :param max_seq_length: Override value for max_seq_length
        """
        sentence_embeddings = model.encode(
            sentences,
            batch_size=args.batch_size,
            show_progress_bar=args.show_progress_bar,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=args.normalize_embeddings,
        )
    elif args.model_type == 'word2vec':
        model = Word2Vec(args.model_name)
        logger.info(f'load model success. model: {model}')
        sentence_embeddings = model.encode(sentences, show_progress_bar=args.show_progress_bar)
    else:
        raise Exception('model_type must be sentencemodel or word2vec')

    logger.info(type(sentence_embeddings), sentence_embeddings.shape)
    # save embeddings to npy file
    np.save(args.output_file, sentence_embeddings)


if __name__ == "__main__":
    main()
