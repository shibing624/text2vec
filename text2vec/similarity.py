# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from enum import Enum
from typing import List

import numpy as np
import torch
from loguru import logger
from numpy import ndarray
from torch import Tensor

from text2vec.sentence_model import SentenceModel, device, EncoderType
from text2vec.utils.distance import cosine_distance
from text2vec.utils.tokenizer import JiebaTokenizer
from text2vec.word2vec import Word2Vec
from text2vec.utils.get_file import deprecated

class SimilarityType(Enum):
    WMD = 0
    COSINE = 1
    # DOT_PRODUCT = 2


class EmbeddingType(Enum):
    WORD2VEC = 0
    BERT = 1


class Similarity:
    def __init__(
            self,
            model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            similarity_type=SimilarityType.COSINE,
            embedding_type=EmbeddingType.BERT,
            encoder_type=EncoderType.MEAN
    ):
        """
        Cal text similarity
        :param model_name_or_path: str, model path or name, default is None,
            auto load paraphrase-multilingual-MiniLM-L12-v2
        :param similarity_type: SimilarityType, similarity type, default is COSINE
        :param embedding_type: EmbeddingType, embedding type, default is BERT
        :param encoder_type: EncoderType, encoder type, adjust with model_name_or_path
        """
        if embedding_type not in [EmbeddingType.BERT, EmbeddingType.WORD2VEC]:
            logger.warning('embedding_type set error, use default bert')
            embedding_type = EmbeddingType.BERT
        if similarity_type not in [SimilarityType.COSINE, SimilarityType.WMD]:
            logger.warning('similarity_type set error, use default cosine')
            similarity_type = SimilarityType.COSINE
        if similarity_type == SimilarityType.WMD and embedding_type != EmbeddingType.WORD2VEC:
            logger.warning('If use wmd sim type, emb type must be w2v')
            embedding_type = EmbeddingType.WORD2VEC
        self.similarity_type = similarity_type
        self.embedding_type = embedding_type
        self.encoder_type = encoder_type
        self.jieba_tokenizer = JiebaTokenizer()

        if self.embedding_type == EmbeddingType.WORD2VEC:
            try:
                # Default: Word2Vec("w2v-light-tencent-chinese")
                self.model = Word2Vec(model_name_or_path)
            except ValueError as e:
                logger.error(f"{model_name_or_path} is not Word2Vec model")
                raise ValueError(e)
        elif self.embedding_type == EmbeddingType.BERT:
            try:
                self.model = SentenceModel(model_name_or_path, encoder_type=self.encoder_type)
            except ValueError as e:
                logger.error(f"{model_name_or_path} is not Bert model")
                raise ValueError(e)
        else:
            raise ValueError('embedding_type error')

    def __str__(self):
        return f"<Similarity> model : {self.model}, similarity_type: {self.similarity_type}, " \
               f"embedding_type: {self.embedding_type}"

    @deprecated("Use similarities instead, `pip install similarities`")
    def get_score(self, sentence1: str, sentence2: str) -> float:
        """
        Get score between text1 and text2
        :param sentence1: str
        :param sentence2: str
        :return: float, score
        """
        res = 0.0
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        if not sentence1 or not sentence2:
            return res
        if self.similarity_type == SimilarityType.COSINE:
            emb1 = self.model.encode(sentence1)
            emb2 = self.model.encode(sentence2)
            res = cos_sim(emb1, emb2)[0] if self.embedding_type == EmbeddingType.BERT else cosine_distance(emb1, emb2)
            res = float(res)
        elif self.similarity_type == SimilarityType.WMD:
            token1 = self.jieba_tokenizer.tokenize(sentence1)
            token2 = self.jieba_tokenizer.tokenize(sentence2)
            res = 1. / (1. + self.model.w2v.wmdistance(token1, token2))
        return res

    @deprecated("Use similarities instead, `pip install similarities`")
    def get_scores(
            self, sentences1: List[str], sentences2: List[str], only_aligned: bool = False
    ) -> ndarray:
        """
        Get similarity scores between sentences1 and sentences2
        :param sentences1: list, sentence1 list
        :param sentences2: list, sentence2 list
        :param only_aligned: bool, default False return all scores, if True only return scores[i][i],
            effective when EmbeddingType.WORD2VEC
        :return: return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not sentences1 or not sentences2:
            return np.array([])
        if only_aligned and len(sentences1) != len(sentences2):
            logger.warning('Sentences size not equal, auto set is_aligned=False')
            only_aligned = False
        embs1 = self.model.encode(sentences1)
        embs2 = self.model.encode(sentences2)
        if self.embedding_type == EmbeddingType.BERT:
            scores = cos_sim(embs1, embs2).numpy()
        else:
            scores = np.zeros((len(sentences1), len(sentences2)), dtype=np.float32)
            if only_aligned:
                for i, e in enumerate(zip(embs1, embs2)):
                    scores[i][i] = cosine_distance(e[0], e[1])
            else:
                for i, e1 in enumerate(embs1):
                    for j, e2 in enumerate(embs2):
                        scores[i][j] = cosine_distance(e1, e2)
        return scores


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def semantic_search(query_embeddings: Tensor,
                    corpus_embeddings: Tensor,
                    query_chunk_size: int = 100,
                    corpus_chunk_size: int = 500000,
                    top_k: int = 10,
                    score_function=cos_sim):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Funtion for computing scores. By default, cosine similarity.
    :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys 'corpus_id' and 'score'
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    query_embeddings = query_embeddings.to(device)
    corpus_embeddings = corpus_embeddings.to(device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarity
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                        corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                       dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    # Sort and strip to top_k results
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    return queries_result_list
