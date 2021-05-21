# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import shutil
import sys
import unittest

sys.path.append('..')
import text2vec
from text2vec.utils.get_file import get_file, hash_file
from text2vec.utils.timer import Timer


class UtilTestCase(unittest.TestCase):
    def test_timer(self):
        timer = Timer()
        start = timer.time
        timer.stop()
        assert timer.time
        timer.resume()
        assert timer.time > start


    def test_get_zip_file(self):
        _url = "https://raw.githubusercontent.com/shibing624/text2vec/master/LICENSE"
        # file_path = get_file(
        #     'train.txt', _url, extract=False,
        #     cache_dir='./',
        #     cache_subdir='./',
        #     verbose=1
        # )
        # print("file_path:", file_path)
        # if os.path.exists(file_path):
        #     shutil.rmtree(file_path)

    def test_jieba(self):
        a = '我要办理特价机票，李浩然可以想办法'
        import jieba
        b = jieba.lcut(a, cut_all=False)
        print('cut_all=False', b)
        self.assertEqual(b, ['我要', '办理', '特价机票', '，', '李', '浩然', '可以', '想', '办法'])

        b = jieba.lcut(a, cut_all=True)
        print('cut_all=True', b)
        self.assertEqual(b, ['我', '要办', '办理', '特价', '特价机票', '机票', '，', '李', '浩然', '可以', '想', '办法'])

        b = jieba.lcut(a, HMM=True)
        print('HMM=True', b)
        self.assertEqual(b, ['我要', '办理', '特价机票', '，', '李', '浩然', '可以', '想', '办法'])

        b = jieba.lcut(a, HMM=False)
        print('HMM=False', b)
        self.assertEqual(b, ['我', '要', '办理', '特价机票', '，', '李', '浩然', '可以', '想', '办法'])


if __name__ == '__main__':
    unittest.main()
