- 基准方法（Word2Vec + Cosine）

尽管文本相似度计算的基准方法很简洁，但用平均词向量之间求余弦相似度的表现非常好。实验有以下结论：

    1. 简单word2vec向量比GloVe向量表现的好
    2. 在用word2vec时，尚不清楚使用停用词表或TF-IDF加权是否更有帮助。在STS数据集上，有一点儿帮助；在SICK上没有帮助。
    仅计算未加权的所有word2vec向量平均值表现得很好。
    3. 在使用GloVe时，停用词列表对于达到好的效果非常重要。利用TF-IDF加权没有帮助。

![基准方法效果很好](https://github.com/shibing624/text2vec/blob/master/docs/base1.jpg)


- 词移距离（WMD）

基于我们的结果，好像没有什么使用词移距离的必要了，因为上述方法表现得已经很好了。只有在STS-TEST数据集上，而且只有在有停止词列表的情况下，词移距离才能和简单基准方法一较高下。

![词移距离的表现令人失望](https://github.com/shibing624/text2vec/blob/master/docs/move1.jpg)
