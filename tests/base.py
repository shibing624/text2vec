# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from simtext import Similarity
sim  = Similarity()
a = '湖北人爱吃鱼'
b = '甘肃人不爱吃鱼'
s = sim.score(a, b)
print(s)