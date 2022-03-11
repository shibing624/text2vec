# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import time
from typing import List, Union, Optional, Dict
from loguru import logger
import numpy as np
from numpy import ndarray
from gensim.models import KeyedVectors
from tqdm import tqdm
from text2vec.utils.get_file import get_file
from text2vec.utils.tokenizer import JiebaTokenizer

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_file = os.path.join(pwd_path, 'stopwords.txt')
USER_DATA_DIR = os.path.expanduser('~/.text2vec/datasets/')


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

    def __init__(self, model_name_or_path: str = 'w2v-light-tencent-chinese',
                 w2v_kwargs: Dict = None,
                 stopwords: List[str] = None,
                 cache_folder: str = USER_DATA_DIR):
        """
        Init word2vec model

        Args:
            model_name_or_path: word2vec file path
            w2v_kwargs: dict, params pass to the ``load_word2vec_format()`` function of ``gensim.models.KeyedVectors`` -
                https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
            stopwords: list, stopwords
            cache_folder: str, save model dir
        """
        self.w2v_kwargs = w2v_kwargs if w2v_kwargs is not None else {}
        if model_name_or_path and os.path.exists(model_name_or_path):
            logger.info('Load pretrained model:{}'.format(model_name_or_path))
        else:
            model_dict = self.model_key_map.get(model_name_or_path, self.model_key_map['w2v-light-tencent-chinese'])
            tar_filename = model_dict.get('tar_filename')
            self.w2v_kwargs.update({'binary': model_dict.get('binary')})
            url = model_dict.get('url')
            untar_filename = model_dict.get('untar_filename')

            # Set new model_name_or_path
            old_model_name = model_name_or_path
            model_name_or_path = os.path.join(cache_folder, untar_filename)
            if old_model_name in self.model_key_map:
                logger.info('Load pretrained model:{}, path:{}'.format(old_model_name, model_name_or_path))
            else:
                logger.warning(f"{old_model_name} not found, Set default model path: {model_name_or_path}")
            if not os.path.exists(model_name_or_path):
                logger.debug(f"Downloading {url} to {model_name_or_path}")
                os.makedirs(cache_folder, exist_ok=True)
                get_file(tar_filename, url, extract=True,
                         cache_dir=cache_folder,
                         cache_subdir=cache_folder,
                         verbose=1)
        t0 = time.time()
        w2v = KeyedVectors.load_word2vec_format(model_name_or_path, **self.w2v_kwargs)
        # w2v.init_sims(replace=True)
        logger.debug('Load w2v from {}, spend {:.2f} sec'.format(model_name_or_path, time.time() - t0))
        self.stopwords = stopwords if stopwords else load_stopwords(default_stopwords_file)
        self.w2v = w2v
        self.jieba_tokenizer = JiebaTokenizer()
        self.model_name_or_path = model_name_or_path

    def __str__(self):
        return f"<Word2Vec, word count: {len(self.w2v.key_to_index)}, emb size: {self.w2v.vector_size}, " \
               f"stopwords count: {len(self.stopwords)}>"

    def encode(self, sentences: Union[List[str], str], show_progress_bar: bool = False) -> ndarray:
        """
        Encode sentences to vectors
        """
        if self.w2v is None:
            raise ValueError('No model for embed sentence')

        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        for sentence in tqdm(sentences, desc='Word2Vec Embeddings', disable=not show_progress_bar):
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
                    ws = self.jieba_tokenizer.tokenize(word, cut_all=True)
                    for w in ws:
                        if w in self.w2v.key_to_index:
                            emb.append(self.w2v.get_vector(w, norm=True))
                            count += 1
            tensor_x = np.array(emb).sum(axis=0)  # 纵轴相加
            if count > 0:
                avg_tensor_x = np.divide(tensor_x, count)
            else:
                avg_tensor_x = np.zeros(self.w2v.vector_size, dtype=float)
            all_embeddings.append(avg_tensor_x)
        all_embeddings = np.array(all_embeddings, dtype=float)
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
