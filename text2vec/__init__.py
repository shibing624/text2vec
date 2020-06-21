# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""

from pathlib import Path

from text2vec.similarity import Similarity, SearchSimilarity, SimType
from text2vec.utils.logger import set_log_level
from text2vec.vector import EmbType, Vector

USER_DIR = Path.expanduser(Path('~')).joinpath('.text2vec')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()

VEC = Vector()
encode = VEC.encode
