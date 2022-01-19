# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), xiaolu(luxiaonlp@163.com)
@description:
"""
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from model import Model
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_helper import CustomDataset, collate_fn, pad_to_maxlen, load_data, load_test_data
import argparse
import scipy.stats

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def set_args():
    """
    参数
    """
    parser = argparse.ArgumentParser('--text2vec进行相似性判断')
    # ./data/ATEC/ATEC.train.data
    # ./data/BQ/BQ.train.data
    # ./data/LCQMC/LCQMC.train.data
    # ./data/PAWSX/PAWSX.train.data
    # ./data/STS-B/STS-B.train.data
    parser.add_argument('--train_path', default='../data/STS-B/STS-B.train.data', type=str, help='训练数据集')
    parser.add_argument('--valid_path', default='../data/STS-B/STS-B.valid.data', type=str, help='验证数据集')
    parser.add_argument('--test_path', default='../data/STS-B/STS-B.test.data', type=str, help='测试数据集')
    parser.add_argument('--pretrained_model_path', default='hfl/chinese-macbert-base', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='训练几轮')
    parser.add_argument('--train_batch_size', default=64, type=int, help='训练批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    return parser.parse_args()


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_sent_id_tensor(tokenizer, s_list):
    input_ids, attention_mask, token_type_ids, = [], [], []
    max_len = max([len(_) + 2 for _ in s_list])
    for s in s_list:
        inputs = tokenizer.encode_plus(text=s, text_pair=None, add_special_tokens=True, return_token_type_ids=True)
        input_ids.append(pad_to_maxlen(inputs['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(inputs['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(inputs['token_type_ids'], max_len=max_len))
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


def evaluate(model, tokenizer, data_path):
    sents1, sents2, labels = load_test_data(data_path)
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    model.to(device)
    model.eval()
    for s1, s2, lab in tqdm(zip(sents1, sents2, labels)):
        input_ids, input_mask, segment_ids = get_sent_id_tensor(tokenizer, [s1, s2])
        lab = torch.tensor([lab], dtype=torch.float)
        input_ids, input_mask, segment_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device)
        lab = lab.to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type='fist-last-avg')
        all_a_vecs.append(output[0].cpu().numpy())
        all_b_vecs.append(output[1].cpu().numpy())
        all_labels.extend(lab.cpu().numpy())
    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    print('sims:', sims[:10])
    print('labels:', all_labels[:10])
    corrcoef = compute_corrcoef(all_labels, sims)
    print('Spearman corr:', corrcoef)
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
    # 3. 奇偶向量相乘
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(y_pred, dim=0)


if __name__ == '__main__':
    args = set_args()
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    # 加载数据集
    train_sentence, train_label = load_data(args.train_path)
    train_dataset = CustomDataset(sentence=train_sentence, label=train_label, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, num_workers=1)
    total_steps = len(train_dataloader) * args.num_train_epochs
    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    model = Model(args.pretrained_model_path)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)
    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % args.train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)
    logs_path = os.path.join(args.output_dir, 'logs.txt')
    for epoch in range(args.num_train_epochs):
        model.train()
        train_label, train_predict = [], []
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            # for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device)
            label_ids = label_ids.to(device)
            output = model(input_ids=input_ids, attention_mask=input_mask, encoder_type='fist-last-avg')
            loss = calc_loss(label_ids, output)
            loss.backward()
            print("当前轮次:{}, 正在迭代:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss))  # 在进度条前面定义一段文字
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            epoch_loss += loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        corr = evaluate(model, tokenizer, args.valid_data)

        with open(logs_path, 'a+') as f:
            s = 'Epoch:{} Valid| corr: {:.6f}\n'.format(epoch, corr)
            f.write(s)
        # 先保存原始transformer bert model
        tokenizer.save_pretrained(args.output_dir)
        model.bert.save_pretrained(args.output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
    model = Model(args.output_dir)
    corr = evaluate(model, tokenizer, args.test_data)
    with open(logs_path, 'a+') as f:
        s = 'Test | corr: {:.6f}\n'.format(corr)
        f.write(s)
