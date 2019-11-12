# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

os.environ['TF_KERAS'] = '1'
from pathlib import Path
import keras_bert

USER_DIR = Path.expanduser(Path('~')).joinpath('.simtext')
if not USER_DIR.exists():
    print('make dir:%s' % USER_DIR)
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()
USER_BERT_MODEL_DIR = USER_DIR.joinpath('bert_model')

custom_objects = keras_bert.get_custom_objects()
