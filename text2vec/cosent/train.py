# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), xiaolu(luxiaonlp@163.com)
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
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
sys.path.append('../..')
from text2vec.cosent.model import Model
from text2vec.cosent.data_helper import CustomDataset, collate_fn, pad_to_maxlen, load_data, load_test_data

pwd_path = os.path.abspath(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_args():
    """
    参数
    """
    parser = argparse.ArgumentParser('--CoSENT进行相似性判断')
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


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


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


def get_sent_id_tensor(tokenizer, sents, max_len):
    input_ids, attention_mask, token_type_ids = [], [], []
    for s in sents:
        inputs = tokenizer.encode_plus(text=s, text_pair=None, add_special_tokens=True, return_token_type_ids=True)
        input_ids.append(pad_to_maxlen(inputs['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(inputs['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(inputs['token_type_ids'], max_len=max_len))
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask, token_type_ids


def evaluate(model, tokenizer, data_path, max_len=64):
    """模型评估函数
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    sents1, sents2, labels = load_test_data(data_path)
    all_labels = []
    source_vecs = []
    target_vecs = []
    model.to(device)
    model.eval()
    for s1, s2, lab in tqdm(zip(sents1, sents2, labels)):
        lab = torch.tensor([lab], dtype=torch.float, device=device)
        all_labels.extend(lab.cpu().numpy())
        input_ids, input_mask, token_type_ids = get_sent_id_tensor(tokenizer, [s1, s2], max_len)
        with torch.no_grad():
            output = model(input_ids, input_mask, token_type_ids)
        source_vecs.append(output[0].cpu().numpy())
        target_vecs.append(output[1].cpu().numpy())
    all_labels = np.array(all_labels)
    source_vecs = np.array(source_vecs)
    target_vecs = np.array(target_vecs)
    # 计算cos相似度，句子向量l2归一化，对应相乘得到
    sims = (l2_normalize(source_vecs) * l2_normalize(target_vecs)).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    logger.debug(f'labels: {all_labels[:10]}')
    logger.debug(f'sims: {sims[:10]}')
    logger.debug(f'Spearman corr: {corrcoef}')
    return corrcoef


def calc_loss(y_true, y_pred):
    """
    CoSENT的排序loss，refer：https://kexue.fm/archives/8847
    """
    # 1. 取出真实的标签
    y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签
    # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_pred = y_pred / norms
    # 3. 奇偶向量相乘, 相似度矩阵除以温度系数0.05(等于*20)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    # 这里加0是因为e^0 = 1相当于在log中加了1
    y_pred = torch.cat((torch.tensor([0]).float().to(device), y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)


if __name__ == '__main__':
    args = set_args()
    logger.info(args)
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    # 加载数据集
    train_sentences, train_labels = load_data(args.train_path)
    # train_sentences, train_labels = train_sentences[:200], train_labels[:200]
    train_dataset = CustomDataset(sentences=train_sentences, labels=train_labels, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, num_workers=1)
    total_steps = len(train_dataloader) * args.num_train_epochs
    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
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
    logger.info("  Num examples = %d" % len(train_dataset))
    logger.info("  Batch size = %d" % args.train_batch_size)
    logger.info("  Num steps = %d" % num_train_optimization_steps)
    logs_path = os.path.join(args.output_dir, 'logs.txt')
    best = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device)
            labels = labels.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = calc_loss(labels, output)
            loss.backward()
            logger.info(f"Epoch:{epoch}, Batch:{step}/{len(train_dataloader)}, Loss:{loss.item():.6f}")
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        corr = evaluate(model, tokenizer, args.valid_path, args.max_len)
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
    corr = evaluate(model, tokenizer, args.test_path, args.max_len)
    with open(logs_path, 'a+') as f:
        f.write(f'Task:{args.task_name}, Test, Spearman corr: {corr:.6f}\n')
