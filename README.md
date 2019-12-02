# text2vec
text2vec, text to vector tool, the most accurate Chinese text similarity calculation tool.(最准的中文文本相似度计算工具)

## Install
* pip3 install text2vec

or

```
git clone https://github.com/shibing624/text2vec.git
cd text2vec
python3 setup.py install
```

## Usage:
```
import text2vec

a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
emb = text2vec.encode(a)
print(emb)
s = text2vec.score(a, b)
print(s)

```

output:
```
0.9569100456524151
```

## Reference
1. [将句子表示为向量（上）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10335164.html)
2. [将句子表示为向量（下）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10341841.html)
3. [《A Simple but Tough-to-Beat Baseline for Sentence Embeddings》[Sanjeev Arora and Yingyu Liang and Tengyu Ma, 2017]](https://openreview.net/forum?id=SyK00v5xx)