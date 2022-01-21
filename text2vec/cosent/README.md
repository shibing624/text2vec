# CoSENT model

CoSENT（Cosine Sentence），比Sentence-BERT更有效的句向量方案

## 实验结果
预训练模型比较了孟子(Mengzi)、MacBERT、BERT等, 只用了训练集训练5个epoch，然后在测试集上给出斯皮尔曼系数（spearman）评测结果。

指定不同数据集，只需在train.py文件中，修改`task_name`参数:  
parser.add_argument('--task_name', default='STS-B', type=str, help='数据集')  

### 本项目实验结果
test测试集的评估结果：

| | ATEC | BQ | LCQMC | PAWSX | STS-B | Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| MacBERT+CoSENT | 50.39 | **72.93** | **79.17** | **60.86** | 79.33 | **68.54**  |
| Mengzi+CoSENT | **50.52** | 72.27 | 78.69 | 12.89 | **80.15** | 58.90 |

### 苏剑林的结果
test测试的评估结果：

| | ATEC | BQ | LCQMC | PAWSX | STS-B | Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BERT+CoSENT | **49.74** | **72.38** | 78.69 | **60.00** | **80.14** | **68.19** |
| Sentence-BERT | 46.36 | 70.36 | **78.72** | 46.86 | 66.41 | 61.74 |
| RoBERTa+CoSENT | **50.81** | **71.45** | **79.31** | **61.56** | **81.13** | **68.85** |
| Sentence-RoBERTa | 48.29 | 69.99 | 79.22 | 44.10 | 72.42 | 62.80 |

### 说明
- 中文预训练模型
以下可以用于中文的预训练语言模型，通过如下方式直接调用transformers下载并使用：
1. MacBERT模型：`--pretrained_model_path hfl/chinese-macbert-base`
2. Mengzi模型：`--pretrained_model_path Langboat/mengzi-bert-base`
3. BERT模型：`--pretrained_model_path bert-base-chinese`
4. RoBERTa模型：`--pretrained_model_path hfl/chinese-roberta-wwm-ext`

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