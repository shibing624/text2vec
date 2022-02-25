# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Rewrite the Base sentence model, add encode function
refer: sentence_transformers.SentenceTransformer and simpletransformers
"""
import os
from enum import Enum
from typing import List, Union, Optional
from tqdm.autonotebook import trange
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


class EncoderType(Enum):
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name


class SentenceModel:
    def __init__(
            self,
            model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encoder_type: EncoderType = EncoderType.MEAN,
            max_seq_length: int = 128
    ):
        """
        Initializes the base sentence model.

        :param model_name_or_path: The name of the model to load from the huggingface models library.
        :param encoder_type: The type of encoder to use, See the EncoderType enum for options:
            FIRST_LAST_AVG, LAST_AVG, CLS, POOLER(cls + dense), MEAN(mean of last_hidden_state)
        :param max_seq_length: The maximum sequence length.

        bert model: https://huggingface.co/transformers/model_doc/bert.html?highlight=bert#transformers.BertModel.forward
        BERT return: <last_hidden_state>, <pooler_output> [hidden_states, attentions]
        Note that: in doc, it says <last_hidden_state> is better semantic summery than <pooler_output>.
        thus, we use <last_hidden_state>.
        """
        self.model_name_or_path = model_name_or_path
        if encoder_type not in list(EncoderType):
            raise ValueError(f"encoder_type must be in {list(EncoderType)}")
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(device)

    def get_sentence_embeddings(self, model_output, attention_mask):
        """
        Returns the model layer output by encoder_type as embeddings.

        Utility function for self.model() method. Not intended to be used directly.
        """
        if self.encoder_type == EncoderType.FIRST_LAST_AVG:
            # Get the first and last hidden states, and average them to get the embeddings
            # hidden_states have 13 list, second is hidden_state
            first = model_output.hidden_states[1]
            last = model_output.hidden_states[-1]
            seq_length = first.size(1)  # Sequence length

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [batch, hid_size]
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [batch, hid_size]
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
                kernel_size=2).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.LAST_AVG:
            sequence_output = model_output.last_hidden_state  # [batch_size, max_len, hidden_size]
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.CLS:
            sequence_output = model_output.last_hidden_state
            return sequence_output[:, 0]  # [batch, hid_size]

        if self.encoder_type == EncoderType.POOLER:
            return model_output.pooler_output  # [batch, hid_size]

        if self.encoder_type == EncoderType.MEAN:
            """
            Mean Pooling - Take attention mask into account for correct averaging
            """
            token_embeddings = model_output.last_hidden_state  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            final_encoding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return final_encoding  # [batch, hid_size]

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32):
        """
        Returns the embeddings for a batch of sentences.
        """
        self.model.eval()
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=True):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            # Tokenize sentences
            inputs = self.tokenizer(sentences_batch, max_length=self.max_seq_length, truncation=True,
                                    padding='max_length', return_tensors='pt')

            input_ids = inputs.get('input_ids').squeeze(1).to(device)
            attention_mask = inputs.get('attention_mask').squeeze(1).to(device)
            token_type_ids = inputs.get('token_type_ids').squeeze(1).to(device)

            # Compute sentences embeddings
            with torch.no_grad():
                embeddings = self.get_sentence_embeddings(
                    self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True),
                    attention_mask
                )
            embeddings = embeddings.detach().cpu()
            all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
