# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

from sentence_transformers import SentenceTransformer


class SBert(SentenceTransformer):
    def __init__(self,
                 model_name_or_path='paraphrase-multilingual-MiniLM-L12-v2',
                 modules=None,
                 device=None,
                 cache_folder=None):
        """
        Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

        :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path,
         it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
         from Huggingface models repository with that name.
        :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can
         be used.
        :param cache_folder: Path to store models
        """
        super(SBert, self).__init__(model_name_or_path, modules, device, cache_folder)
