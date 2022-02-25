# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: This examples trains CoSENT model with the English NLI dataset, and predict in STS benchmark dataset.
It generates sentence embeddings that can be compared using cosine-similarity to measure the similarity.
"""
import argparse
import sys
import csv
import gzip
import time
import numpy as np
from loguru import logger
import os

sys.path.append('..')
from text2vec import CosentModel, compute_spearmanr, EncoderType, CosentTrainDataset, CosentTestDataset
from text2vec import cos_sim, http_get


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


def load_en_stsb_dataset(stsb_file):
    # Convert the dataset to a DataLoader ready for training
    logger.info("Read STSbenchmark dataset")
    train_samples = []
    valid_samples = []
    test_samples = []
    with gzip.open(stsb_file, 'rt', encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score'])
            if row['split'] == 'dev':
                valid_samples.append((row['sentence1'], row['sentence2'], score))
            elif row['split'] == 'test':
                test_samples.append((row['sentence1'], row['sentence2'], score))
            else:
                score = int(score > 2.5)
                train_samples.append((row['sentence1'], row['sentence2'], score))
    return train_samples, valid_samples, test_samples


def main():
    parser = argparse.ArgumentParser('CoSENT Text Matching task')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='name of transformers model')
    parser.add_argument('--stsb_file', default='data/English-STS-B/stsbenchmark.tsv.gz', type=str,
                        help='Train data path')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument('--output_dir', default='./outputs/STS-B-en-cosent-unsup', type=str,
                        help='Model output directory')
    parser.add_argument('--max_seq_length', default=64, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--nli_limit_size', default=200000, type=float, help='Learning rate')
    parser.add_argument('--encoder_type', default='FIRST_LAST_AVG', type=lambda t: EncoderType[t],
                        choices=list(EncoderType), help='Encoder type, string name of EncoderType')
    args = parser.parse_args()
    logger.info(args)

    _, valid_samples, test_samples = load_en_stsb_dataset(args.stsb_file)

    if args.do_train:
        model = CosentModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                            max_seq_length=args.max_seq_length)
        # 无监督实验，在NLI train数据上训练，评估模型在STS test上的表现
        data_dir = os.path.abspath(os.path.dirname(args.stsb_file))
        nli_dataset_path = os.path.join(data_dir, 'AllNLI.tsv.gz')
        if not os.path.exists(nli_dataset_path):
            http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
        # Read the AllNLI.tsv.gz file and create the training dataset
        logger.info("Read AllNLI train dataset")
        nli_train_samples = []
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'train':
                    label_id = label2int[row['label']]
                    nli_train_samples.append((row['sentence1'], label_id))
                    nli_train_samples.append((row['sentence2'], label_id))
                    if len(nli_train_samples) > args.nli_limit_size:
                        break

        train_dataset = CosentTrainDataset(model.tokenizer, nli_train_samples, args.max_seq_length)
        valid_dataset = CosentTestDataset(model.tokenizer, valid_samples, args.max_seq_length)

        model.train(train_dataset,
                    args.output_dir,
                    eval_dataset=valid_dataset,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    lr=args.learning_rate)
        logger.info(f"Model saved to {args.output_dir}")

    if args.do_predict:
        model = CosentModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                            max_seq_length=args.max_seq_length)
        # Predict embeddings
        srcs = []
        trgs = []
        labels = []
        for terms in test_samples:
            src, trg, label = terms[0], terms[1], terms[2]
            srcs.append(src)
            trgs.append(trg)
            labels.append(label)
        logger.debug(f'{test_samples[0]}')
        sentence_embeddings = model.encode(srcs)
        logger.debug(f"{type(sentence_embeddings)}, {sentence_embeddings.shape}, {sentence_embeddings[0].shape}")
        # Predict similarity scores
        calc_similarity_scores(args, srcs, trgs, labels)


if __name__ == '__main__':
    main()
