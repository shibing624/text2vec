# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pathlib import Path
from text2vec.similarity import Similarity

USER_DIR = Path.expanduser(Path('~')).joinpath('.text2vec')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()


SIM = Similarity()
score = SIM.score
encode = SIM.encode
