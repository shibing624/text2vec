# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
import sys
import random
import numpy as np
import argparse
import scipy.stats
from tqdm import tqdm
from loguru import logger
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

sys.path.append('../..')
from text2vec.sentence_bert.model import Model
from text2vec.sentence_bert.data_helper import TrainDataset, load_data

pwd_path = os.path.abspath(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_args():
    """
    参数
    """
    parser = argparse.ArgumentParser('--Sentence-BERT进行相似性判断')
    # ./data/ATEC/ATEC.train.data
    # ./data/BQ/BQ.train.data
    # ./data/LCQMC/LCQMC.train.data
    # ./data/PAWSX/PAWSX.train.data
    # ./data/STS-B/STS-B.train.data
    parser.add_argument('--task_name', default='STS-B', type=str, help='数据名称')
    parser.add_argument('--train_path', default=os.path.join(pwd_path, '../data/STS-B/STS-B.train.data'), type=str,
                        help='训练数据集')
    parser.add_argument('--valid_path', default=os.path.join(pwd_path, '../data/STS-B/STS-B.valid.data'), type=str,
                        help='验证数据集')
    parser.add_argument('--test_path', default=os.path.join(pwd_path, '../data/STS-B/STS-B.test.data'), type=str,
                        help='测试数据集')
    parser.add_argument('--pretrained_model_path', default='hfl/chinese-macbert-base', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出')
    parser.add_argument('--max_len', default=64, type=int, help='句子最大长度')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练几轮')
    parser.add_argument('--num_classes', default=2, type=int, help='类别数')
    parser.add_argument('--train_batch_size', default=64, type=int, help='训练批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    args = parser.parse_args()
    if args.task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']:
        args.train_path = os.path.join(pwd_path, f'../data/{args.task_name}/{args.task_name}.train.data')
        args.valid_path = os.path.join(pwd_path, f'../data/{args.task_name}/{args.task_name}.valid.data')
        args.test_path = os.path.join(pwd_path, f'../data/{args.task_name}/{args.task_name}.test.data')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(vecs):
    """
    L2标准化
    """
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """
    Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def compute_pearsonr(x, y):
    """
    Pearson系数
    """
    return scipy.stats.perasonr(x, y)[0]


def evaluate(model, dataloader):
    """模型评估函数
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    all_labels = []
    all_preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for source, target, labels in tqdm(dataloader):
            labels = labels.to(device)
            all_labels.extend(labels.cpu().numpy())
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            outputs = model(source_input_ids, source_attention_mask, source_token_type_ids,
                            target_input_ids, target_attention_mask, target_token_type_ids, is_train=False)
            preds = torch.cosine_similarity(outputs[0], outputs[1])
            all_preds.extend(preds.cpu().numpy())
    corrcoef = compute_corrcoef(all_labels, all_preds)
    logger.debug(f'labels: {all_labels[:10]}')
    logger.debug(f'preds:  {all_preds[:10]}')
    logger.debug(f'Spearman corr: {corrcoef}')
    return corrcoef


def calc_loss(y_pred, y_true):
    """
    Calc loss with two sentence embeddings
    """
    loss = nn.CrossEntropyLoss()(y_pred, y_true)
    return loss


if __name__ == '__main__':
    args = set_args()
    logger.info(args)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    model = Model(args.pretrained_model_path, encoder_type='first-last-avg', num_classes=args.num_classes)
    model.to(device)

    # 加载数据集
    train_data = load_data(args.train_path)
    # train_data = train_data[:200]
    train_dataset = TrainDataset(train_data, tokenizer=tokenizer, max_len=args.max_len)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.train_batch_size)
    valid_dataloader = DataLoader(
        dataset=TrainDataset(load_data(args.valid_path, is_train=False), tokenizer, args.max_len),
        batch_size=args.train_batch_size)
    test_dataloader = DataLoader(
        dataset=TrainDataset(load_data(args.test_path, is_train=False), tokenizer, args.max_len),
        batch_size=args.train_batch_size)
    total_steps = len(train_dataloader) * args.num_train_epochs
    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

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
    logger.info("  Num examples = %d" % len(train_dataset))
    logger.info("  Batch size = %d" % args.train_batch_size)
    logger.info("  Num steps = %d" % num_train_optimization_steps)
    logs_path = os.path.join(args.output_dir, 'logs.txt')
    best = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            source, target, labels = batch
            labels = labels.to(device)
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            outputs = model(source_input_ids, source_attention_mask, source_token_type_ids,
                            target_input_ids, target_attention_mask, target_token_type_ids)
            loss = calc_loss(outputs, labels)
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
