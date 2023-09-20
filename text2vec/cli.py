# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cli for text2vec
"""
import argparse
import sys

import pandas as pd
from loguru import logger

sys.path.append('..')
from text2vec.sentence_model import SentenceModel
from text2vec.word2vec import Word2Vec


def save_partial_results(df, output_file, is_first_chunk):
    mode = 'w' if is_first_chunk else 'a'
    header = is_first_chunk

    with open(output_file, mode, encoding='utf-8') as f:
        df.to_csv(f, index=False, header=header)


def main():
    parser = argparse.ArgumentParser(description='text2vec cli')
    parser.add_argument('--input_file', type=str, help='input file path, text file, required', required=True)
    parser.add_argument('--output_file', type=str, default='text_embs.csv', help='output file path, output csv file')
    parser.add_argument('--model_type', type=str, default='sentencemodel', help='model type: sentencemodel, word2vec')
    parser.add_argument('--model_name', type=str, default='shibing624/text2vec-base-chinese', help='model name or path')
    parser.add_argument('--encoder_type', type=str, default='MEAN',
                        help='encoder type: MEAN, CLS, POOLER, FIRST_LAST_AVG, LAST_AVG')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
    parser.add_argument('--chunk_size', type=int, default=1000, help='chunk size to save partial results')
    parser.add_argument('--device', type=str, default=None, help='device: cpu, cuda')
    parser.add_argument('--show_progress_bar', type=bool, default=True, help='show progress bar, default True')
    parser.add_argument('--normalize_embeddings', type=bool, default=False, help='normalize embeddings, default False')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='multi gpu, default False')
    args = parser.parse_args()
    logger.debug(args)

    sentences = set()
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.add(line)
    sentences = list(sentences)
    logger.info(f'load sentences success. sentences num: {len(sentences)}, top3: {sentences[:3]}')

    if args.model_type == 'sentencemodel':
        model = SentenceModel(
            model_name_or_path=args.model_name,
            encoder_type=args.encoder_type,
            max_seq_length=args.max_seq_length,
            device=args.device,
        )
    elif args.model_type == 'word2vec':
        model = Word2Vec(args.model_name)
    else:
        raise ValueError('model_type must be sentencemodel or word2vec')
    logger.info(f'load model success. model: {model}')

    if args.multi_gpu:
        if args.model_type == 'word2vec':
            raise ValueError('word2vec model not support multi gpu')
        pool = model.start_multi_process_pool()
        # Compute the embeddings using the multi processes pool
        embeddings = model.encode_multi_process(
            sentences,
            pool,
            batch_size=args.batch_size,
            normalize_embeddings=args.normalize_embeddings
        )
        # Optional: Stop the process in the pool
        model.stop_multi_process_pool(pool)
        df = pd.DataFrame({'sentence': sentences, 'sentence_embeddings': embeddings.tolist()})
        save_partial_results(df, args.output_file, True)
        logger.debug(f'saved results. size: {len(sentences)}, emb shape: {embeddings.shape}')
    else:
        chunk_size = args.chunk_size
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]

            if args.model_type == 'sentencemodel':
                chunk_embeddings = model.encode(
                    chunk_sentences,
                    batch_size=args.batch_size,
                    show_progress_bar=args.show_progress_bar,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=args.normalize_embeddings,
                )
            elif args.model_type == 'word2vec':
                chunk_embeddings = model.encode(
                    chunk_sentences,
                    show_progress_bar=args.show_progress_bar
                )
            else:
                raise ValueError('model_type must be sentencemodel or word2vec')

            # Save part embeddings to dataframe
            chunk_df = pd.DataFrame({'sentence': chunk_sentences, 'sentence_embeddings': chunk_embeddings.tolist()})
            save_partial_results(chunk_df, args.output_file, i == 0)
            logger.debug(f'saved partial results. size: {len(chunk_sentences)}, emb shape: {chunk_embeddings.shape}')
    logger.info(f"Input file {args.input_file}, saved embeddings to {args.output_file} success.")


if __name__ == "__main__":
    main()
