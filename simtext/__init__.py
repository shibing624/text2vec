# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pathlib import Path

USER_DIR = Path.expanduser(Path('~')).joinpath('.simtext')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()
USER_TUNED_MODELS_DIR = USER_DIR.joinpath('tuned_models')
