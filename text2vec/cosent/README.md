# CoSENT model

CoSENT（Cosine Sentence），比Sentence-BERT更有效的句向量方案

## 实验结果
预训练模型比较了孟子(Mengzi)、MacBERT、BERT等, 只用了训练集训练5个epoch，然后在测试集上给出斯皮尔曼系数（spearman）评测结果。

指定不同数据集，只需在train.py文件中，修改`task_name`参数:  
parser.add_argument('--task_name', default='STS-B', type=str, help='数据集')  

### 中文匹配数据集测评

| Model Name | ATEC | BQ | LCQMC | PAWSX | STS-B | Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| MacBERT+CoSENT | 50.39 | **72.93** | **79.17** | **60.86** | **80.51** | **68.77**  |
| Mengzi+CoSENT | **50.52** | 72.27 | 78.69 | 12.89 | 80.15 | 58.90 |
| BERT+CoSENT | **49.74** | **72.38** | 78.69 | **60.00** | **80.14** | **68.19** |
| Sentence-BERT | 46.36 | 70.36 | **78.72** | 46.86 | 66.41 | 61.74 |
| RoBERTa+CoSENT | **50.81** | **71.45** | **79.31** | **61.56** | **81.13** | **68.85** |
| Sentence-RoBERTa | 48.29 | 69.99 | 79.22 | 44.10 | 72.42 | 62.80 |

### 英文匹配数据集测评
| Arch | Backbone | Model Name | English-STS-B | 
| :-: | :-: | :-: | :-: |
| BERT | bert-base-uncased | BERT-base-cls | 20.29 |
| BERT | bert-base-uncased | BERT-base-first_last_avg | 59.04 |
| BERT | bert-base-uncased | BERT-base-first_last_avg-whiten(NLI) | 63.65 |
| SBERT | bert-base-nli-mean-tokens | SBERT-base-nli-cls | 73.65 |
| SBERT | bert-base-nli-mean-tokens | SBERT-base-nli-first_last_avg | 77.96 |
| CoSENT | bert-base-uncased | CoSENT-base-first_last_avg | 69.93 |
| CoSENT | bert-base-nli-mean-tokens | CoSENT-base-nli-first_last_avg | 79.68 |

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

- 复现脚本
执行以下脚本，直接复现上表中`MacBERT+CoSENT`的模型效果：
```shell
cd cosent
sh train.sh 
```
## 使用说明
#### 训练
```shell
python train.py
```
#### 预测
```shell
python infer.py
```

# Reference
- CoSENT-keras: https://github.com/bojone/CoSENT
- CoSENT原理：https://kexue.fm/archives/8847
- 孟子预训练模型: https://github.com/Langboat/Mengzi