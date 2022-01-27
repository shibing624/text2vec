# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark
from scratch. It generates sentence embeddings that can be compared using cosine-similarity to measure the
similarity.
"""
import csv
import gzip
import os
import sys
import argparse
from tqdm import tqdm
from loguru import logger
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

sys.path.append('../..')
from text2vec.cosent.model import Model
from text2vec.cosent.train import set_seed, calc_loss, evaluate
from text2vec.cosent.data_helper import TrainDataset, TestDataset
from text2vec.utils.get_file import http_get

pwd_path = os.path.abspath(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_args():
    """
    参数
    """
    parser = argparse.ArgumentParser('--CoSENT进行相似性判断')
    parser.add_argument('--task_name', default='English-STS-B', type=str, help='数据名称')
    parser.add_argument('--is_nli_and_stsb', default=False, type=bool, help='Trained on NLI data + STSb data')
    # 'sentence-transformers/bert-base-nli-mean-tokens'
    parser.add_argument('--pretrained_model_path', default='bert-base-uncased', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='./outputs/train_on_english_stsb', type=str, help='模型输出')
    parser.add_argument('--max_len', default=64, type=int, help='句子最大长度')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练几轮')
    parser.add_argument('--train_batch_size', default=64, type=int, help='训练批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    args = parser.parse_args()
    return args


def train(model, train_dataloader, valid_dataloader, test_dataloader, args, tokenizer):
    total_steps = len(train_dataloader) * args.num_train_epochs
    num_train_optimization_steps = int(
        len(train_dataloader.dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d" % len(train_dataloader.dataset))
    logger.info("  Batch size = %d" % args.train_batch_size)
    logger.info("  Num steps = %d" % num_train_optimization_steps)
    logs_path = os.path.join(args.output_dir, 'logs.txt')
    best = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            source, labels = batch
            labels = labels.to(device)
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            outputs = model(source_input_ids, source_attention_mask, source_token_type_ids)
            loss = calc_loss(labels, outputs)
            loss.backward()
            logger.info(f"Epoch:{epoch}, Batch:{step}/{len(train_dataloader)}, Loss:{loss.item():.6f}")

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        corr = evaluate(model, valid_dataloader)
        with open(logs_path, 'a+') as f:
            f.write(f'Task:{args.task_name}, Epoch:{epoch}, Valid, Spearman corr: {corr:.6f}\n')
        model.train()
        if best < corr:
            best = corr
            # 先保存原始transformer bert model
            tokenizer.save_pretrained(args.output_dir)
            model.bert.save_pretrained(args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info(f"higher corrcoef: {best:.6f} in epoch: {epoch}, save model")
    model = Model(args.output_dir, encoder_type='first-last-avg')
    corr = evaluate(model, test_dataloader)
    with open(logs_path, 'a+') as f:
        f.write(f'Task:{args.task_name}, Test, Spearman corr: {corr:.6f}\n')


def main():
    args = set_args()
    # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    logger.info(args)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    model = Model(args.pretrained_model_path, encoder_type='first-last-avg')
    model.to(device)

    # Check if dataset exsist. If not, download and extract it
    sts_dataset_path = os.path.join(pwd_path, '../data/English-STS-B/', 'stsbenchmark.tsv.gz')
    if not os.path.exists(sts_dataset_path):
        http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    # Convert the dataset to a DataLoader ready for training
    logger.info("Read STSbenchmark train dataset")
    train_samples = []
    valid_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as f:
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
    sts_train_dataloader = DataLoader(dataset=TrainDataset(train_samples, tokenizer, args.max_len), shuffle=True,
                                      batch_size=args.train_batch_size)
    valid_dataloader = DataLoader(dataset=TestDataset(valid_samples, tokenizer, args.max_len),
                                  batch_size=args.train_batch_size)
    test_dataloader = DataLoader(dataset=TestDataset(test_samples, tokenizer, args.max_len),
                                 batch_size=args.train_batch_size)
    if args.is_nli_and_stsb:
        nli_dataset_path = os.path.join(pwd_path, '../data/English-STS-B/', 'AllNLI.tsv.gz')
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
        nli_train_dataloader = DataLoader(dataset=TrainDataset(nli_train_samples, tokenizer, args.max_len),
                                          shuffle=True, batch_size=args.train_batch_size)
        # 无监督实验，在NLI train数据上训练，评估模型在STS test上的表现
        train(model, nli_train_dataloader, valid_dataloader, test_dataloader, args, tokenizer)
    else:
        # 有监督实验，在STS train数据上训练，评估模型在STS test上的表现
        train(model, sts_train_dataloader, valid_dataloader, test_dataloader, args, tokenizer)


if __name__ == '__main__':
    main()
