# text2vec
[![PyPI version](https://badge.fury.io/py/text2vec.svg)](https://badge.fury.io/py/text2vec)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/text2vec.svg)](https://github.com/shibing624/text2vec/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/text2vec.svg)](https://github.com/shibing624/text2vec/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

text2vec, chinese text to vector.(文本向量化表示工具，包括：词向量化表示，句子向量化表示，长文本向量化表示，文本相似度计算。)


# Feature
#### 文本向量表示
- 字词粒度，通过腾讯AI Lab开源的大规模高质量中文[词向量数据（800万中文词轻量版）](https://pan.baidu.com/s/1La4U4XNFe8s5BJqxPQpeiQ) (文件名：light_Tencent_AILab_ChineseEmbedding.bin 密码: tawe），获取字词的word2vec向量表示。
- 句子粒度，通过求句子中所有单词词向量的平均值计算得到。
- 篇章粒度，可以通过gensim库的doc2vec得到，应用较少，本项目不实现。

#### 文本相似度计算

- 基准方法，估计两句子间语义相似度最简单的方法就是求句子中所有单词词向量的平均值，然后计算两句子词向量之间的余弦相似性。
- 词移距离（Word Mover’s Distance），词移距离使用两文本间的词向量，测量其中一文本中的单词在语义空间中移动到另一文本单词所需要的最短距离。

#### query和docs的相似度比较
- rank_bm25方法，使用bm25的变种算法，对query和文档之间的相似度打分，得到docs的rank排序。

## 调研结论

#### 文本相似度计算
- 基准方法

尽管文本相似度计算的基准方法很简洁，但用平均词向量之间求余弦相似度的表现非常好。实验有以下结论：

    1. 简单word2vec向量比GloVe向量表现的好
    2. 在用word2vec时，尚不清楚使用停用词表或TF-IDF加权是否更有帮助。在STS数据集上，有一点儿帮助；在SICK上没有帮助。
    仅计算未加权的所有word2vec向量平均值表现得很好。
    3. 在使用GloVe时，停用词列表对于达到好的效果非常重要。利用TF-IDF加权没有帮助。

![基准方法效果很好](./docs/base1.jpg)


- 词移距离

基于我们的结果，好像没有什么使用词移距离的必要了，因为上述方法表现得已经很好了。只有在STS-TEST数据集上，而且只有在有停止词列表的情况下，词移距离才能和简单基准方法一较高下。

![词移距离的表现令人失望](./docs/move1.jpg)

# Demo

http://42.193.145.218/product/short_text_sim/

# Install
```
pip3 install text2vec
```

or

```
git clone https://github.com/shibing624/text2vec.git
cd text2vec
python3 setup.py install
```

# Usage

1. 下载预训练词向量文件

以下词向量，任选一个：

- 轻量版腾讯词向量 [百度云盘-密码:tawe](https://pan.baidu.com/s/1La4U4XNFe8s5BJqxPQpeiQ) 或 [谷歌云盘](https://drive.google.com/u/0/uc?id=1iQo9tBb2NgFOBxx0fA16AZpSgc-bG_Rp&export=download)，二进制，111MB放到 `~/.text2vec/datasets/light_Tencent_AILab_ChineseEmbedding.bin`

- [腾讯词向量-官方全量](https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz), 6.78G放到： `~/.text2vec/datasets/Tencent_AILab_ChineseEmbedding.txt`


2. 文本向量计算

- 基于预训练的word2vec计算文本向量

通过腾讯词向量计算各字词的词向量，句子向量通过单词词向量取平均值得到。

示例[base_demo.py](./examples/base_demo.py)

```python
import text2vec
# 计算字向量
char = '卡'
emb = text2vec.encode(char)
print(type(emb), emb.shape)

# 计算词向量
word = '银行卡'
print(word, text2vec.encode(word))

# 计算句子向量
a = '如何更换花呗绑定银行卡'
emb = text2vec.encode(a)
print(a, emb)
```

output:
```
<class 'numpy.ndarray'> (200,)

银行卡 [ 0.0020064  -0.12582362  ...     0.09727262]

如何更换花呗绑定银行卡 [ 0.0412493  -0.12568748  ...  0.02760466]

```
> 返回值`emb`是`numpy.ndarray`类型，shape为`(200, )`


- 计算句子之间的相似度值

示例[similarity_demo.py](./examples/similarity_demo.py)

```
from text2vec import Similarity

a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'

sim = Similarity()
s = sim.get_score(a, b)
print(s)

```

output:
```
0.9551
```

> 句子相似度值范围在0到1之间，值越大越相似。

- 计算句子与文档集之间的相似度值

一般在文档候选集中找与query最相似的文本，常用于QA场景的问句相似匹配任务。


```
from text2vec import SearchSimilarity

a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
c = '我什么时候开通了花呗'

corpus = [a, b, c]
print(corpus)
search_sim = SearchSimilarity(corpus=corpus)

print(a, 'scores:', search_sim.get_scores(query=a))
print(a, 'rank similarities:', search_sim.get_similarities(query=a))
```

output:
```
['如何更换花呗绑定银行卡', '花呗更改绑定银行卡', '我什么时候开通了花呗']
如何更换花呗绑定银行卡 scores: [ 0.9527457  -0.07449248 -0.03204909]
如何更换花呗绑定银行卡 rank similarities: ['如何更换花呗绑定银行卡', '我什么时候开通了花呗', '花呗更改绑定银行卡']
```

> 'search_sim.get_scores(query)'的结果越大，表示该query与corpus的相似度越近。

> 'search_sim.get_similarities(query)'的结果表示corpus中所有句子与query相似度rank排序的结果，越靠前的结果越相似。


- 基于BERT预训练模型的文本向量计算

基于中文BERT预训练模型`chinese_L-12_H-768_A-12(bert-base-chinese)`，提取后四层layers计算各token的词向量，句子向量通过单词词向量取平均值得到。

示例[bert_emb_demo.py](examples/bert_emb_demo.py)

```python
from text2vec import Vector
from text2vec import Similarity

vec = Vector(embedding_type='bert')
char = '卡'
emb = vec.encode(char)
# <class 'numpy.ndarray'> (128, 3072) 128=seq_len, 3072=768*4
print(type(emb), emb.shape)

word = '银行卡'
print(word, vec.encode(word))

a = '如何更换花呗绑定银行卡'
emb = vec.encode(a)
print(a, emb)
print(emb.shape)

sim = Similarity(embedding_type='bert')
b = '花呗更改绑定银行卡'
print(sim.get_score(a, b))
```
output:
```
<class 'numpy.ndarray'> (128, 3072)

银行卡 [[-0.91093725  0.54408133 -0.4109965  ...  0.48191142 -0.48503038
   0.26117468]
 ...
 [-1.2602367   0.00573288 -0.6756776  ...  0.33792442 -0.1214454
   0.11303894]]

如何更换花呗绑定银行卡 [[-0.6816437   0.13298336  0.11106233 ... -0.4509677  -0.4271722
  -0.39778918]
 ...
 [-0.5474275  -0.24083692 -0.6185864  ... -0.20258519 -0.2466043
  -0.2537103 ]]

(128, 3072)

0.9087
```

# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/text2vec.svg)](https://github.com/shibing624/text2vec/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：个人名称-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Cite

如果你在研究中使用了text2vec，请按如下格式引用：

```latex
@software{text2vec,
  author = {Xu Ming},
  title = {text2vec: A Tool for Text to Vector},
  year = {2021},
  url = {https://github.com/shibing624/text2vec},
}
```

# License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加text2vec的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python setup.py test`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

# Reference

1. [将句子表示为向量（上）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10335164.html)
2. [将句子表示为向量（下）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10341841.html)
3. [A Simple but Tough-to-Beat Baseline for Sentence Embeddings[Sanjeev Arora and Yingyu Liang and Tengyu Ma, 2017]](https://openreview.net/forum?id=SyK00v5xx)
4. [四种计算文本相似度的方法对比[Yves Peirsman]](https://zhuanlan.zhihu.com/p/37104535)
5. [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)
