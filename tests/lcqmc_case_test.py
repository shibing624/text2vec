# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from typing import List

import simtext

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


def apply_bert_case(cases: List[List[str]]):
    for line in cases:
        q1 = line[0]
        q2 = line[1]
        a = line[2]

        s = simtext.score(q1, q2)
        print(f'q1:{q1}, q2:{q2}, expect a:{a}, actual a:{s}')


w2v_sim = simtext.Similarity(embedding_type='w2v')


def apply_w2v_case(cases: List[List[str]]):
    for line in cases:
        q1 = line[0]
        q2 = line[1]
        a = line[2]

        s = w2v_sim.score(q1, q2)
        print(f'q1:{q1}, q2:{q2}, expect a:{a}, actual a:{s}')


def test_case1():
    apply_bert_case(case_same_keywords)
    # q1:飞行员没钱买房怎么办？, q2:父母没钱买房子, expect a:False, actual a:0.846145250083127
    # q1:聊天室都有哪些好的, q2:聊天室哪个好, expect a:True, actual a:0.9302744928458128
    # q1:不锈钢上贴的膜怎么去除, q2:不锈钢上的胶怎么去除, expect a:True, actual a:0.9625103602739326
    # q1:动漫人物的口头禅, q2:白羊座的动漫人物, expect a:False, actual a:0.9254731623309143
    apply_w2v_case(case_same_keywords)
    # q1:飞行员没钱买房怎么办？, q2:父母没钱买房子, expect a:False, actual a:0.3761319521207672
    # q1:聊天室都有哪些好的, q2:聊天室哪个好, expect a:True, actual a:0.6761800308242283
    # q1:不锈钢上贴的膜怎么去除, q2:不锈钢上的胶怎么去除, expect a:True, actual a:0.9680308625170728
    # q1:动漫人物的口头禅, q2:白羊座的动漫人物, expect a:False, actual a:1.0


def test_case2():
    apply_bert_case(case_categories_corresponding_pairs)
    # q1:从广州到长沙在哪里定高铁票, q2:在长沙哪里坐高铁回广州？, expect a:False, actual a:0.9243307258941305
    # q1:请问现在最好用的听音乐软件是什么啊, q2:听歌用什么软件比较好, expect a:True, actual a:0.8689272207970375
    # q1:谁有吃过完美的产品吗？如何？, q2:完美产品好不好, expect a:True, actual a:0.8615442355066881
    # q1:朱熹是哪个朝代的诗人, q2:朱熹是明理学的集大成者，他生活在哪个朝代, expect a:True, actual a:0.8630224628595393
    # q1:这是哪个奥特曼？, q2:这是什么奥特曼..., expect a:True, actual a:0.9050183814431813
    # q1:网上找工作可靠吗, q2:网上找工作靠谱吗, expect a:True, actual a:0.9935955342946918
    # q1:你们都喜欢火影忍者里的谁啊, q2:火影忍者里你最喜欢谁, expect a:True, actual a:0.8950941719662809
    apply_w2v_case(case_categories_corresponding_pairs)
    # q1:从广州到长沙在哪里定高铁票, q2:在长沙哪里坐高铁回广州？, expect a:False, actual a:0.7925367285111807
    # q1:请问现在最好用的听音乐软件是什么啊, q2:听歌用什么软件比较好, expect a:True, actual a:0.7905707175554252
    # q1:谁有吃过完美的产品吗？如何？, q2:完美产品好不好, expect a:True, actual a:0.5775977787974922
    # q1:朱熹是哪个朝代的诗人, q2:朱熹是明理学的集大成者，他生活在哪个朝代, expect a:True, actual a:0.7512713030942898
    # q1:这是哪个奥特曼？, q2:这是什么奥特曼..., expect a:True, actual a:0.5318252085174675
    # q1:网上找工作可靠吗, q2:网上找工作靠谱吗, expect a:True, actual a:0.9507644087770979
    # q1:你们都喜欢火影忍者里的谁啊, q2:火影忍者里你最喜欢谁, expect a:True, actual a:0.8474923584324268
