"""
@file   : run_sentence_bert_transformers.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-02
"""
import os
import gzip
import pickle
import torch
import time
import numpy as np
from tqdm import tqdm
from torch import nn
from config import set_args
from model import SentenceBert
from torch.utils.data import TensorDataset, DataLoader
from utils import compute_corrcoef, l2_normalize
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup


class Features:
    def __init__(self, s1_input_ids=None, s2_input_ids=None, label=None):
        self.s1_input_ids = s1_input_ids
        self.s2_input_ids = s2_input_ids
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "s1_input_ids: %s" % (self.s1_input_ids)
        s += ", s2_input_ids: %s" % (self.s2_input_ids)
        s += ", label: %d" % (self.label)
        return s


def convert_token_to_id(path, tokenizer, max_len=64):
    '''
    将句子转为id序列
    :return:
    '''
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        features = []
        for line in tqdm(lines):
            s, t, lab = line.strip().split('\t')
            s_input = tokenizer.encode(s)
            t_input = tokenizer.encode(t)
            
            if len(s_input) > max_len:
                s_input = s_input[:max_len]
            else:
                s_input = s_input + (max_len - len(s_input)) * [0]
            

            if len(t_input) > max_len:
                t_input = t_input[:max_len]
            else:
                t_input = t_input + (max_len - len(t_input)) * [0]
            lab = int(lab)
            feature = Features(s1_input_ids=s_input, s2_input_ids=t_input, label=lab)
            features.append(feature)
        return features

def evaluate(model):
    model.eval()
    # 语料向量化
    all_a_vecs, all_b_vecs = [], []
    all_labels = []
    for step, batch in tqdm(enumerate(test_dataloader)):
        s1_input_ids, s2_input_ids, label = batch
        if torch.cuda.is_available():
            s1_input_ids = s1_input_ids.cuda()
            s2_input_ids = s2_input_ids.cuda()
            label = label.cuda()
        with torch.no_grad():
            s1_embeddings, s2_embeddings = model(s1_input_ids=s1_input_ids, s2_input_ids=s2_input_ids)
            s1_embeddings = s1_embeddings.cpu().numpy()
            s2_embeddings = s2_embeddings.cpu().numpy()
            label = label.cpu().numpy()
           
            all_a_vecs.extend(s1_embeddings)
            all_b_vecs.extend(s2_embeddings)
            all_labels.extend(label)

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    return corrcoef


def calc_loss(s1_vec, s2_vec, true_label):
    loss_fct = nn.MSELoss()
    output = torch.cosine_similarity(s1_vec, s2_vec)
    loss = loss_fct(output, true_label)
    return loss


if __name__ == '__main__':
    args = set_args()
    args.output_dir = 'outputs'
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据集
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    max_len = 64
    train_features = convert_token_to_id(args.train_data, tokenizer, max_len)
    test_features = convert_token_to_id(args.test_data, tokenizer, max_len)

    # 开始训练
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_features)))
    print("  Batch size = {}".format(args.train_batch_size))
    train_s1_input_ids = torch.tensor([f.s1_input_ids for f in train_features], dtype=torch.long)
    train_s2_input_ids = torch.tensor([f.s2_input_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.float32)
    train_data = TensorDataset(train_s1_input_ids, train_s2_input_ids, train_label_ids)
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    
    test_s1_input_ids = torch.tensor([f.s1_input_ids for f in test_features], dtype=torch.long)
    test_s2_input_ids = torch.tensor([f.s2_input_ids for f in test_features], dtype=torch.long)
    test_label_ids = torch.tensor([f.label for f in test_features], dtype=torch.float32)
    test_data = TensorDataset(test_s1_input_ids, test_s2_input_ids, test_label_ids)
    test_dataloader = DataLoader(test_data, batch_size=args.train_batch_size, shuffle=True)

    num_train_steps = int(
         len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # 模型
    model = SentenceBert(model_name=args.pretrained_model_path)

    # 指定gpu运行
    if torch.cuda.is_available():
        model = model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = 0.05 * num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
            s1_input_ids, s2_input_ids, label = batch
            s1_vec, s2_vec = model(s1_input_ids, s2_input_ids)

            loss = calc_loss(s1_vec, s2_vec, label)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            print('Epoch:{}, Step:{}, Loss:{:10f}, Time:{:10f}'.format(epoch, step, loss, time.time() - start_time))
            loss.backward()

            # nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)   # 是否进行梯度裁剪

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # global_step += 1
                # corrcoef = evaluate(model)

        # 一轮跑完 进行eval
        corrcoef = evaluate(model)
        ss = 'epoch:{}, corrcoef:{}'.format(epoch, corrcoef)
        with open(args.output_dir + '/logs.txt', 'a+', encoding='utf8') as f:
            ss += '\n'
            f.write(ss)
        best_corrcoef = None
        if best_corrcoef is None or best_corrcoef < corrcoef:
            best_corrcoef = corrcoef
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "best_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)


