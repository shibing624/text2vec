# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

from text2vec.similarity import Similarity, SearchSimilarity, SimType
from text2vec.utils.logger import set_log_level
from text2vec.vector import EmbType, Vector
from text2vec.version import __version__

USER_DATA_DIR = os.path.expanduser('~/.text2vec/datasets/')
os.makedirs(USER_DATA_DIR, exist_ok=True)

VEC = Vector()
encode = VEC.encode
set_stopwords_file = VEC.set_stopwords_file
