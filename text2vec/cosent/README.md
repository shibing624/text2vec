# CoSENT model

CoSENT（Cosine Sentence），比Sentence-BERT更有效的句向量方案

## 实验结果
预训练模型比较了孟子(Mengzi)、MacBERT、BERT等, 只用了训练集训练5个epoch，然后在测试集上给出斯皮尔曼系数（spearman）评测结果。

指定不同数据集，只需在train.py文件中，修改`task_name`参数:  
parser.add_argument('--task_name', default='STS-B', type=str, help='数据集')  

- 英文匹配数据集的评测结果：

| Arch | Backbone | Model Name | English-STS-B | 
| :-: | :-: | :-: | :-: |
| GloVe | glove | Avg_word_embeddings_glove_6B_300d | 61.77 |
| BERT | bert-base-uncased | BERT-base-cls | 20.29 |
| BERT | bert-base-uncased | BERT-base-first_last_avg | 59.04 |
| BERT | bert-base-uncased | BERT-base-first_last_avg-whiten(NLI) | 63.65 |
| SBERT | sentence-transformers/bert-base-nli-mean-tokens | SBERT-base-nli-cls | 73.65 |
| SBERT | sentence-transformers/bert-base-nli-mean-tokens | SBERT-base-nli-first_last_avg | 77.96 |
| SBERT | xlm-roberta-base | paraphrase-multilingual-MiniLM-L12-v2 | 84.42 |
| CoSENT | bert-base-uncased | CoSENT-base-first_last_avg | 69.93 |
| CoSENT | sentence-transformers/bert-base-nli-mean-tokens | CoSENT-base-nli-first_last_avg | 79.68 |

- 中文匹配数据集的评测结果：

| Arch | Backbone | Model Name | ATEC | BQ | LCQMC | PAWSX | STS-B | Avg | QPS |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| CoSENT | hfl/chinese-macbert-base | CoSENT-macbert-base | 50.39 | **72.93** | **79.17** | **60.86** | **80.51** | **68.77**  | 2572 |
| CoSENT | Langboat/mengzi-bert-base | CoSENT-mengzi-base | **50.52** | 72.27 | 78.69 | 12.89 | 80.15 | 58.90 | 2502 |
| CoSENT | bert-base-chinese | CoSENT-bert-base | 49.74 | 72.38 | 78.69 | 60.00 | 80.14 | 68.19 | 2653 |
| SBERT | bert-base-chinese | SBERT-bert-base | 46.36 | 70.36 | 78.72 | 46.86 | 66.41 | 61.74 | 1365 |
| SBERT | hfl/chinese-macbert-base | SBERT-macbert-base | 47.28 | 68.63 | **79.42** | 55.59 | 64.82 | 63.15 | 1948 |
| CoSENT | hfl/chinese-roberta-wwm-ext | CoSENT-roberta-ext | **50.81** | **71.45** | **79.31** | **61.56** | **81.13** | **68.85** | - |
| SBERT | hfl/chinese-roberta-wwm-ext | SBERT-roberta-ext | 48.29 | 69.99 | 79.22 | 44.10 | 72.42 | 62.80 | - |


### 说明
- 中文预训练模型
以下可以用于中文的预训练语言模型，通过如下方式直接调用transformers下载并使用：
1. MacBERT模型：`--pretrained_model_path hfl/chinese-macbert-base`
2. Mengzi模型：`--pretrained_model_path Langboat/mengzi-bert-base`
3. BERT模型：`--pretrained_model_path bert-base-chinese`
4. RoBERTa模型：`--pretrained_model_path hfl/chinese-roberta-wwm-ext`

- SBERT指Sentence-BERT
- bert-base-nli-mean-tokens是`--pretrained_model_path sentence-transformers/bert-base-nli-mean-tokens`
- 以上结果均只用该数据集的train训练，在test上评估得到的结果，没用外部数据
- 中文匹配模型的pooling方法都是用的`first_last_avg`

- 复现脚本
执行以下脚本，直接复现上表中`MacBERT+CoSENT`的模型效果：


```shell
cd examples
CUDA_VISIBLE_DEVICES=0
python3 training_sup_cosent.py --do_train --do_predict --num_epochs 10 --output_dir outputs/STS-B-model > outputs/STS-B.log 2>&1
```

## 使用说明
#### 训练
```shell
python3 training_sup_cosent.py --do_train --num_epochs 10 --output_dir outputs/STS-B-model > outputs/STS-B.log 2>&1
```
#### 预测
```shell
python3 training_sup_cosent.py --do_predict --output_dir outputs/STS-B-model > outputs/STS-B.log 2>&1
```

# Reference
- CoSENT-keras: https://github.com/bojone/CoSENT
- CoSENT原理：https://kexue.fm/archives/8847
- 孟子预训练模型: https://github.com/Langboat/Mengzi