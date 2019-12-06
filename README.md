# text2vec
text2vec, chinese text to vetor.(文本向量化表示工具，包括词向量化、句子向量化)



## Feature
#### 文本向量表示
- 字词粒度，通过腾讯AI Lab开源的大规模高质量中文[词向量数据（800万中文词）](https://ai.tencent.com/ailab/nlp/embedding.html)，获取字词的word2vec向量表示。
- 句子粒度，通过求句子中所有单词词嵌入的平均值计算得到。
- 篇章粒度，可以通过gensim库的doc2vec得到，应用较少，本项目不实现。

#### 文本相似度计算

- 基准方法，估计两句子间语义相似度最简单的方法就是求句子中所有单词词嵌入的平均值，然后计算两句子词嵌入之间的余弦相似性。
- 词移距离（Word Mover’s Distance），词移距离使用两文本间的词嵌入，测量其中一文本中的单词在语义空间中移动到另一文本单词所需要的最短距离。

#### query和docs的相似度比较
- rank_bm25方法，使用bm25的变种算法，对query和文档之间的相似度打分，得到docs的rank排序。

## Result

#### 文本相似度计算
- 基准方法

尽管文本相似度计算的基准方法很简洁，但用平均词嵌入之间求余弦相似度的表现非常好。实验有以下结论：

    1. 简单word2vec嵌入比GloVe嵌入表现的好
    2. 在用word2vec时，尚不清楚使用停用词表或TF-IDF加权是否更有帮助。在STS数据集上，有一点儿帮助；在SICK上没有帮助。仅计算未加权的所有word2vec嵌入平均值表现得很好。
    3. 在使用GloVe时，停用词列表对于达到好的效果非常重要。利用TF-IDF加权没有帮助。

![基准方法效果很好](./docs/base1.jpg)


- 词移距离

基于我们的结果，好像没有什么使用词移距离的必要了，因为上述方法表现得已经很好了。只有在STS-TEST数据集上，而且只有在有停止词列表的情况下，词移距离才能和简单基准方法一较高下。

![词移距离的表现令人失望](./docs/move1.jpg)



## Install
```
pip3 install text2vec
```

or

```
git clone https://github.com/shibing624/text2vec.git
cd text2vec
python3 setup.py install
```

## Usage:
- get text vector
```

import text2vec

char = '我'
print(char, text2vec.encode(char))

word = '如何'
print(word, text2vec.encode(word))

a = '如何更换花呗绑定银行卡'
emb = text2vec.encode(a)
print(a, emb)

```


- get similarity score between text1 and text2

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
0.9519710685638405
```


- get text similarity score between query and docs

```

from text2vec import SearchSimilarity

a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
c = '我什么时候开通了花呗'

corpus = [a, b, c]
search_sim = SearchSimilarity(corpus=corpus)

print(search_sim.get_scores(query=a))
print(search_sim.get_similarities(query=a))
```


## Reference

1. [将句子表示为向量（上）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10335164.html)
2. [将句子表示为向量（下）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10341841.html)
3. [A Simple but Tough-to-Beat Baseline for Sentence Embeddings[Sanjeev Arora and Yingyu Liang and Tengyu Ma, 2017]](https://openreview.net/forum?id=SyK00v5xx)
4. [四种计算文本相似度的方法对比[Yves Peirsman]](https://zhuanlan.zhihu.com/p/37104535)
5. [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)