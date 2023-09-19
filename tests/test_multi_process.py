# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 

code copy from: SentenceTransformers.tests.test_multi_process.py
"""

import sys
import unittest

sys.path.append('..')
from text2vec import SentenceModel
import numpy as np


class ComputeMultiProcessTest(unittest.TestCase):
    def setUp(self):
        self.model = SentenceModel()

    def test_multi_gpu_encode(self):
        # Start the multi processes pool on all available CUDA devices
        pool = self.model.start_multi_process_pool(['cpu', 'cpu'])

        sentences = ["This is sentence {}".format(i) for i in range(1000)]

        # Compute the embeddings using the multi processes pool
        emb = self.model.encode_multi_process(sentences, pool, chunk_size=50)
        assert emb.shape == (len(sentences), 768)

        emb_normal = self.model.encode(sentences)

        diff = np.max(np.abs(emb - emb_normal))
        print("Max multi proc diff", diff)
        assert diff < 0.001
