# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import os
import shutil

import simtext
from simtext import utils


def timer_test():
    timer = utils.Timer()
    start = timer.time
    timer.stop()
    assert timer.time
    timer.resume()
    assert timer.time > start


def get_file_test():
    _url = "https://raw.githubusercontent.com/shibing624/simtext/master/LICENSE"
    file_path = utils.get_file(
        'LICENSE', _url, extract=True,
        cache_dir=simtext.USER_DATA_DIR,
        cache_subdir='LICENSE',
        verbose=1
    )
    print("file_path:", file_path)
    num_lines = 201
    assert len(open(file_path, 'rb').readlines()) == num_lines
    file_hash = utils.hash_file(file_path, algorithm='md5')

    file_path2 = utils.get_file(
        'LICENSE', _url, extract=False,
        md5_hash=file_hash,
        cache_dir=simtext.USER_DATA_DIR,
        cache_subdir='LICENSE',
        verbose=1
    )
    file_hash2 = utils.hash_file(file_path2, algorithm='md5')
    assert file_hash == file_hash2

    file_dir = simtext.USER_DATA_DIR.joinpath('LICENSE')
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)


def get_zip_file():
    _url = "https://raw.githubusercontent.com/shibing624/simtext/master/tests/a_dir.zip"
    file_path = utils.get_file(
        'aa', _url, extract=True,
        cache_dir=simtext.USER_DATA_DIR,
        cache_subdir='bert_test',
        verbose=1
    )
    print("file_path:", file_path)


if __name__ == '__main__':
    get_zip_file()
