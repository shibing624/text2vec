"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-02
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--使用transformers实现sentence_bert')
    parser.add_argument('--train_data', default='../data/STS-B/STS-B.train.data', type=str, help='训练数据集')
    parser.add_argument('--test_data', default='../data/STS-B/STS-B.test.data', type=str, help='测试数据集')
    parser.add_argument('--pretrained_model_path', default='bert-base-chinese', type=str, help='预训练模型的路径')
    parser.add_argument('--train_batch_size', default=64, type=int, help='训练批次的大小')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出目录')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚的大小')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练几轮')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')

    return parser.parse_args()
