# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')

from text2vec import SentenceModel, cos_sim

sbert_model = SentenceModel()

# query1 query2 matching?
case_same_keywords = [['飞行员没钱买房怎么办？', '父母没钱买房子', False],
                      ['聊天室都有哪些好的', '聊天室哪个好', True],
                      ['不锈钢上贴的膜怎么去除', '不锈钢上的胶怎么去除', True],
                      ['动漫人物的口头禅', '白羊座的动漫人物', False]]

case_categories_corresponding_pairs = [['从广州到长沙在哪里定高铁票', '在长沙哪里坐高铁回广州？', False],
                                       ['请问现在最好用的听音乐软件是什么啊', '听歌用什么软件比较好', True],
                                       ['谁有吃过完美的产品吗？如何？', '完美产品好不好', True],
                                       ['朱熹是哪个朝代的诗人', '朱熹是明理学的集大成者，他生活在哪个朝代', True],
                                       ['这是哪个奥特曼？', '这是什么奥特曼...', True],
                                       ['网上找工作可靠吗', '网上找工作靠谱吗', True],
                                       ['你们都喜欢火影忍者里的谁啊', '火影忍者里你最喜欢谁', True]]


def sbert_sim_score(str_a, str_b):
    a_emb = sbert_model.encode(str_a)
    b_emb = sbert_model.encode(str_b)
    return cos_sim(a_emb, b_emb).item()


def apply_sbert_case(cases):
    for line in cases:
        q1 = line[0]
        q2 = line[1]
        a = line[2]

        s = sbert_sim_score(q1, q2)
        print(f'q1:{q1}, q2:{q2}, expect:{a}, actual:{s:.4f}')


class LcqTestCase(unittest.TestCase):
    def test_sbert(self):
        """测试sbert结果"""
        apply_sbert_case(case_same_keywords)
        apply_sbert_case(case_categories_corresponding_pairs)
        # q1: 飞行员没钱买房怎么办？, q2: 父母没钱买房子, expect: False, actual: 0.3742
        # q1: 聊天室都有哪些好的, q2: 聊天室哪个好, expect: True, actual: 0.9497
        # q1: 不锈钢上贴的膜怎么去除, q2: 不锈钢上的胶怎么去除, expect: True, actual: 0.8708
        # q1: 动漫人物的口头禅, q2: 白羊座的动漫人物, expect: False, actual: 0.8510
        # q1: 从广州到长沙在哪里定高铁票, q2: 在长沙哪里坐高铁回广州？, expect: False, actual: 0.9163
        # q1: 请问现在最好用的听音乐软件是什么啊, q2: 听歌用什么软件比较好, expect: True, actual: 0.9182
        # q1: 谁有吃过完美的产品吗？如何？, q2: 完美产品好不好, expect: True, actual: 0.7370
        # q1: 朱熹是哪个朝代的诗人, q2: 朱熹是明理学的集大成者，他生活在哪个朝代, expect: True, actual: 0.7382
        # q1: 这是哪个奥特曼？, q2: 这是什么奥特曼..., expect: True, actual: 0.8744
        # q1: 网上找工作可靠吗, q2: 网上找工作靠谱吗, expect: True, actual: 0.9531
        # q1: 你们都喜欢火影忍者里的谁啊, q2: 火影忍者里你最喜欢谁, expect: True, actual: 0.9643

if __name__ == '__main__':
    unittest.main()
