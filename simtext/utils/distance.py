# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import numpy as np


def cos_dist(emb_1, emb_2):
    """
    calc cos distance
    return cos score
    """
    num = float(np.sum(emb_1 * emb_2))
    denom = np.linalg.norm(emb_1) * np.linalg.norm(emb_2)
    cos = num / denom if denom > 0 else 0.0
    return cos
