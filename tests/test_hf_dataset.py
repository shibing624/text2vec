# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')

from datasets import load_dataset


class DatasetTestCase(unittest.TestCase):

    def test_data_diff(self):
        test_dataset = load_dataset("shibing624/nli_zh", "STS-B", split="test")

        # Predict embeddings
        srcs = []
        trgs = []
        labels = []
        for terms in test_dataset:
            src, trg, label = terms['sentence1'], terms['sentence2'], terms['label']
            srcs.append(src)
            trgs.append(trg)
            labels.append(label)
            if len(src) > 100:
                break
        print(f'{test_dataset[0]}')
        print(f'{srcs[0]}')


if __name__ == '__main__':
    unittest.main()
