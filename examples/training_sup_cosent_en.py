# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: This examples trains CoSENT model with the English STS dataset.
It generates sentence embeddings that can be compared using cosine-similarity to measure the similarity.
"""
import argparse
import sys
import csv
import gzip
import time
import numpy as np
from loguru import logger

sys.path.append('..')
from text2vec.cosent.cosent_model import CosentModel, compute_spearmanr, EncoderType
from text2vec.cosent.cosent_dataset import CosentTrainDataset, CosentTestDataset
from text2vec import cos_sim


def calc_similarity_scores(args, sents1, sents2, labels):
    m = CosentModel(args.output_dir, encoder_type=EncoderType.FIRST_LAST_AVG, max_seq_length=args.max_seq_length)
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
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='name of transformers model')
    parser.add_argument('--stsb_file', default='data/English-STS-B/stsbenchmark.tsv.gz', type=str,
                        help='Train data path')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument('--output_dir', default='./outputs/STS-B-en-cosent', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=64, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    args = parser.parse_args()
    logger.info(args)

    test_dataset = []
    if args.do_train:
        model = CosentModel(model_name_or_path=args.model_name, encoder_type=EncoderType.FIRST_LAST_AVG,
                            max_seq_length=args.max_seq_length)

        # Convert the dataset to a DataLoader ready for training
        logger.info("Read STSbenchmark dataset")
        train_samples = []
        valid_samples = []
        test_samples = []
        with gzip.open(args.stsb_file, 'rt', encoding='utf8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score'])
                if row['split'] == 'dev':
                    valid_samples.append((row['sentence1'], row['sentence2'], score))
                elif row['split'] == 'test':
                    test_samples.append((row['sentence1'], row['sentence2'], score))
                else:
                    train_samples.append((row['sentence1'], score))
                    train_samples.append((row['sentence2'], score))
        train_dataset = CosentTrainDataset(model.tokenizer, train_samples, args.max_seq_length)
        valid_dataset = CosentTestDataset(model.tokenizer, valid_samples, args.max_seq_length)
        test_dataset = CosentTestDataset(model.tokenizer, test_samples, args.max_seq_length)
        model.train(train_dataset,
                    args.output_dir,
                    eval_dataset=valid_dataset,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    lr=args.learning_rate)
        logger.info(f"Model saved to {args.output_dir}")
    if args.do_predict:
        model = CosentModel(model_name_or_path=args.output_dir, encoder_type=EncoderType.FIRST_LAST_AVG,
                            max_seq_length=args.max_seq_length)
        test_data = test_dataset
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
