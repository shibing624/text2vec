# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from typing import List, Union
from loguru import logger
import numpy as np
from text2vec.utils.get_file import get_file


class NGram:
    def __init__(self, model_name_or_path=None, cache_folder=os.path.expanduser('~/.pycorrector/datasets/')):
        if model_name_or_path and os.path.exists(model_name_or_path):
            logger.info('Load kenlm language model:{}'.format(model_name_or_path))
            language_model_path = model_name_or_path
        else:
            # 语言模型 2.95GB
            get_file(
                'zh_giga.no_cna_cmn.prune01244.klm',
                'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
                extract=True,
                cache_subdir=cache_folder,
                verbose=1)
            language_model_path = os.path.join(cache_folder, 'zh_giga.no_cna_cmn.prune01244.klm')
        try:
            import kenlm
        except ImportError:
            raise ImportError('Kenlm not installed, use "pip install kenlm".')
        self.lm = kenlm.Model(language_model_path)
        logger.debug('Loaded language model: %s.' % language_model_path)

    def ngram_score(self, sentence: str):
        """
        取n元文法得分
        :param sentence: str, 输入的句子
        :return:
        """
        return self.lm.score(' '.join(sentence), bos=False, eos=False)

    def perplexity(self, sentence: str):
        """
        取语言模型困惑度得分，越小句子越通顺
        :param sentence: str, 输入的句子
        :return:
        """
        return self.lm.perplexity(' '.join(sentence))

    def encode(self, sentences: Union[List[str], str]):
        """
        将句子转换成ngram特征向量
        """
        if self.lm is None:
            raise ValueError('No model for embed sentence')

        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        for sentence in sentences:
            ngram_avg_scores = []
            for n in [2, 3]:
                scores = []
                for i in range(len(sentence) - n + 1):
                    word = sentence[i:i + n]
                    score = self.ngram_score(word)
                    scores.append(score)
                if scores:
                    # 移动窗口补全得分
                    for _ in range(n - 1):
                        scores.insert(0, scores[0])
                        scores.append(scores[-1])
                    avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(sentence))]
                else:
                    avg_scores = np.zeros(len(sentence), dtype=float)
                ngram_avg_scores.append(avg_scores)
            # 取拼接后的n-gram平均得分
            sent_scores = np.average(np.array(ngram_avg_scores), axis=0)
            all_embeddings.append(sent_scores)
        all_embeddings = np.asarray(all_embeddings, dtype=object)
        if input_is_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings
