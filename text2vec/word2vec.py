# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import time

import numpy as np
from gensim.models import KeyedVectors

from text2vec.utils.get_file import get_file
from text2vec.utils.log import logger
from text2vec.utils.tokenizer import Tokenizer

USER_DATA_DIR = os.path.expanduser('~/.text2vec/datasets/')
os.makedirs(USER_DATA_DIR, exist_ok=True)

pwd_path = os.path.abspath(os.path.dirname(__file__))


def load_stopwords(file_path):
    stopwords = set()
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                stopwords.add(line)
    return stopwords


class Word2Vec:
    """Pre-trained word2vec embedding"""
    model_key_map = {
        # 腾讯词向量, 6.78G
        'w2v-tencent-chinese': {
            'tar_filename': 'Tencent_AILab_ChineseEmbedding.tar.gz',
            'url': 'https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz',
            'binary': False,
            'untar_filename': 'Tencent_AILab_ChineseEmbedding.txt'},
        # 轻量版腾讯词向量，二进制，111MB
        'w2v-light-tencent-chinese': {
            'tar_filename': 'light_Tencent_AILab_ChineseEmbedding.bin',
            'url': 'http://42.193.145.218/light_Tencent_AILab_ChineseEmbedding.bin',
            'binary': True,
            'untar_filename': 'light_Tencent_AILab_ChineseEmbedding.bin'},
    }

    def __init__(self, model_name_or_path='w2v-light-tencent-chinese',
                 w2v_kwargs={},
                 stopwords_file=os.path.join(pwd_path, 'data/stopwords.txt'),
                 cache_folder=USER_DATA_DIR):
        """
        Init word2vec model

        Args:
            model_name_or_path: word2vec file path
            w2v_kwargs: params pass to the ``load_word2vec_format()`` function of ``gensim.models.KeyedVectors`` -
                https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
            stopwords_file: str, stopwords
            cache_folder: str, save model
        """
        self.w2v_kwargs = w2v_kwargs
        if model_name_or_path and os.path.exists(model_name_or_path):
            logger.info('Load pretrained model:{}'.format(model_name_or_path))
            model_path = model_name_or_path
        else:
            model_dict = self.model_key_map.get(model_name_or_path, self.model_key_map['w2v-light-tencent-chinese'])
            tar_filename = model_dict.get('tar_filename')
            self.w2v_kwargs.update({'binary': model_dict.get('binary')})
            url = model_dict.get('url')
            untar_filename = model_dict.get('untar_filename')
            model_path = os.path.join(cache_folder, untar_filename)
            if not os.path.exists(model_path):
                get_file(tar_filename, url, extract=True,
                         cache_dir=USER_DATA_DIR,
                         cache_subdir=USER_DATA_DIR,
                         verbose=1)
        t0 = time.time()
        w2v = KeyedVectors.load_word2vec_format(model_path, **self.w2v_kwargs)
        # w2v.init_sims(replace=True)
        logger.debug('Load w2v from {}, spend {:.2f} sec'.format(model_name_or_path, time.time() - t0))
        self.stopwords = load_stopwords(stopwords_file)
        self.w2v = w2v
        self.tokenizer = Tokenizer()
        logger.debug('Word count: {}, emb size: {}'.format(len(w2v.key_to_index), w2v.vector_size))
        logger.debug('Set stopwords: {}, count: {}'.format(sorted(list(self.stopwords))[:10], len(self.stopwords)))

    def encode(self, sentences):
        """
        Encode embed sentences

        Args:
            sentences: Sentence list to embed
        Returns:
            vectorized sentence list
        """
        if self.w2v is None:
            raise ValueError('need to build model for embed sentence')

        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        for sentence in sentences:
            emb = []
            count = 0
            for word in sentence:
                # 过滤停用词
                if word in self.stopwords:
                    continue
                # 调用词向量
                if word in self.w2v.key_to_index:
                    emb.append(self.w2v.get_vector(word, norm=True))
                    count += 1
                else:
                    if len(word) == 1:
                        continue
                    # 再切分，eg 特价机票
                    ws = self.tokenizer.tokenize(word, cut_all=True)
                    for w in ws:
                        if w in self.w2v.key_to_index:
                            emb.append(self.w2v.get_vector(w, norm=True))
                            count += 1
            tensor_x = np.array(emb).sum(axis=0)  # 纵轴相加
            if count > 0:
                avg_tensor_x = np.divide(tensor_x, count)
            else:
                avg_tensor_x = 0.0
            all_embeddings.append(avg_tensor_x)
        all_embeddings = np.array(all_embeddings)
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
