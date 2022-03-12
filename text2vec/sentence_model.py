# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Base sentence model function, add encode function.
Parts of this file is adapted from the sentence-transformers library at https://github.com/UKPLab/sentence-transformers.
"""
import os
from enum import Enum
from typing import List, Union, Optional
from tqdm.autonotebook import trange
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from text2vec.utils.stats_util import compute_spearmanr, compute_pearsonr

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

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()


class SentenceModel:
    def __init__(
            self,
            model_name_or_path: str = "shibing624/text2vec-base-chinese",
            encoder_type: Union[str, EncoderType] = "MEAN",
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
        encoder_type = EncoderType.from_string(encoder_type) if isinstance(encoder_type, str) else encoder_type
        if encoder_type not in list(EncoderType):
            raise ValueError(f"encoder_type must be in {list(EncoderType)}")
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.bert.to(device)
        self.results = {}  # Save training process evaluation result

    def __str__(self):
        return f"<SentenceModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}>"

    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids):
        """
        Returns the model output by encoder_type as embeddings.

        Utility function for self.bert() method.
        """
        model_output = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

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
            token_embeddings = model_output.last_hidden_state  # Contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            final_encoding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return final_encoding  # [batch, hid_size]

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, show_progress_bar: bool = False):
        """
        Returns the embeddings for a batch of sentences.
        """
        self.bert.eval()
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            # Tokenize sentences
            inputs = self.tokenizer(sentences_batch, max_length=self.max_seq_length, truncation=True,
                                    padding='max_length', return_tensors='pt')
            input_ids = inputs.get('input_ids').squeeze(1).to(device)
            attention_mask = inputs.get('attention_mask').squeeze(1).to(device)
            token_type_ids = inputs.get('token_type_ids').squeeze(1).to(device)

            # Compute sentences embeddings
            with torch.no_grad():
                embeddings = self.get_sentence_embeddings(input_ids, attention_mask, token_type_ids)
            embeddings = embeddings.detach().cpu()
            all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def eval_model(self, eval_dataset: Dataset, output_dir: str = None, verbose: bool = True, batch_size: int = 16):
        """
        Evaluates the model on eval_df. Saves results to args.output_dir
            result: Dictionary containing evaluation results.
        """
        result = self.evaluate(eval_dataset, output_dir, batch_size=batch_size)
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result

    def evaluate(self, eval_dataset, output_dir: str = None, batch_size: int = 16):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        results = {}

        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.bert.to(device)
        self.bert.eval()

        batch_labels = []
        batch_preds = []
        for batch in tqdm(eval_dataloader, disable=False, desc="Running Evaluation"):
            source, target, labels = batch
            labels = labels.to(device)
            batch_labels.extend(labels.cpu().numpy())
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)

            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)

            with torch.no_grad():
                source_embeddings = self.get_sentence_embeddings(source_input_ids, source_attention_mask,
                                                                 source_token_type_ids)
                target_embeddings = self.get_sentence_embeddings(target_input_ids, target_attention_mask,
                                                                 target_token_type_ids)
                preds = torch.cosine_similarity(source_embeddings, target_embeddings)
            batch_preds.extend(preds.cpu().numpy())

        spearman = compute_spearmanr(batch_labels, batch_preds)
        pearson = compute_pearsonr(batch_labels, batch_preds)
        logger.debug(f"labels: {batch_labels[:10]}")
        logger.debug(f"preds:  {batch_preds[:10]}")
        logger.debug(f"pearson: {pearson}, spearman: {spearman}")

        results["eval_spearman"] = spearman
        results["eval_pearson"] = pearson
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "eval_results.txt"), "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def save_model(self, output_dir, model, results=None):
        """
        Saves the model to output_dir.
        :param output_dir:
        :param model:
        :param results:
        :return:
        """
        logger.info(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))
