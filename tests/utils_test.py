# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""

import os
import shutil

import text2vec
from text2vec.utils.get_file import get_file, hash_file
from text2vec.utils.timer import Timer


def test_timer():
    timer = Timer()
    start = timer.time
    timer.stop()
    assert timer.time
    timer.resume()
    assert timer.time > start


def test_get_file():
    _url = "https://raw.githubusercontent.com/shibing624/text2vec/master/LICENSE"
    file_path = get_file(
        'LICENSE', _url, extract=True,
        cache_dir=text2vec.USER_DATA_DIR,
        cache_subdir='LICENSE',
        verbose=1
    )
    print("file_path:", file_path)
    num_lines = 201
    assert len(open(file_path, 'rb').readlines()) == num_lines
    file_hash = hash_file(file_path, algorithm='md5')

    file_path2 = get_file(
        'LICENSE', _url, extract=False,
        md5_hash=file_hash,
        cache_dir=text2vec.USER_DATA_DIR,
        cache_subdir='LICENSE',
        verbose=1
    )
    file_hash2 = hash_file(file_path2, algorithm='md5')
    assert file_hash == file_hash2

    file_dir = text2vec.USER_DATA_DIR.joinpath('LICENSE')
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)


def test_get_zip_file():
    _url = "https://raw.githubusercontent.com/pengming617/bert_textMatching/master/data/train.txt"
    file_path = get_file(
        'train.txt', _url, extract=True,
        cache_dir='./',
        cache_subdir='./',
        verbose=1
    )
    print("file_path:", file_path)
    if os.path.exists(file_path):
        shutil.rmtree(file_path)


def test_bin_file():
    _url = 'https://www.borntowin.cn/mm/emb_models/sentence_w2v.bin'
    file_path = get_file(
        'sentence_w2v.bin', _url, extract=True,
        cache_dir=text2vec.USER_DIR,
        cache_subdir=text2vec.USER_DATA_DIR
    )
    print("file_path:", file_path)
    if os.path.exists(file_path):
        shutil.rmtree(file_path)


def test_jieba():
    a = '我要办理特价机票，李浩然可以想办法'
    import jieba
    b = jieba.lcut(a, cut_all=False)
    print('cut_all=False', b)
    b = jieba.lcut(a, cut_all=True)
    print('cut_all=True', b)

    b = jieba.lcut(a, HMM=True)
    print('HMM=True', b)

    b = jieba.lcut(a, HMM=False)
    print('HMM=False', b)
