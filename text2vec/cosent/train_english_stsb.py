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
from tqdm import tqdm
from loguru import logger
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

sys.path.append('../..')
from text2vec.cosent.model import Model
from text2vec.cosent.train import set_args, set_seed, calc_loss, evaluate
from text2vec.cosent.data_helper import TrainDataset, TestDataset

pwd_path = os.path.abspath(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    args = set_args()
    # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    args.pretrained_model_path = 'bert-base-uncased'  # 'sentence-transformers/bert-base-nli-mean-tokens'
    args.output_dir = './outputs/train_english_stsb'
    logger.info(args)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    # Check if dataset exsist. If not, download and extract it
    sts_dataset_path = os.path.join(pwd_path, '../data/English-STS-B/stsbenchmark.tsv.gz')
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

    train_dataloader = DataLoader(dataset=TrainDataset(train_samples, tokenizer, args.max_len), shuffle=True,
                                  batch_size=args.train_batch_size)
    valid_dataloader = DataLoader(dataset=TestDataset(valid_samples, tokenizer, args.max_len),
                                  batch_size=args.train_batch_size)
    test_dataloader = DataLoader(dataset=TestDataset(test_samples, tokenizer, args.max_len),
                                 batch_size=args.train_batch_size)

    total_steps = len(train_dataloader) * args.num_train_epochs
    num_train_optimization_steps = int(
        len(train_dataloader) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    model = Model(args.pretrained_model_path, encoder_type='first-last-avg')
    model.to(device)
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
    logger.info("  Num examples = %d" % len(train_dataloader))
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
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

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


if __name__ == '__main__':
    main()
