# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Base sentence model function, add encode function.
Parts of this file is adapted from the sentence-transformers: https://github.com/UKPLab/sentence-transformers
"""
import math
import os
import queue
from enum import Enum
from typing import List, Union, Optional, Dict

import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.autonotebook import trange
from transformers import AutoTokenizer, AutoModel

from text2vec.utils.stats_util import compute_spearmanr, compute_pearsonr

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
            encoder_type: str = "MEAN",
            max_seq_length: int = 256,
            device: Optional[str] = None,
    ):
        """
        Initializes the base sentence model.

        :param model_name_or_path: The name of the model to load from the huggingface models library.
        :param encoder_type: The type of encoder to use, See the EncoderType enum for options:
            FIRST_LAST_AVG, LAST_AVG, CLS, POOLER(cls + dense), MEAN(mean of last_hidden_state)
        :param max_seq_length: The maximum sequence length.
        :param device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation.
            If None, checks if a GPU can be used.
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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            logger.debug("Use pytorch device: {}".format(device))
        self.device = torch.device(device)
        self.bert.to(self.device)
        self.results = {}  # Save training process evaluation result

    def __str__(self):
        return (f"<SentenceModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, "
                f"max_seq_length: {self.max_seq_length}, emb_dim: {self.get_sentence_embedding_dimension()}")

    def get_sentence_embedding_dimension(self):
        """
        Get the dimension of the sentence embeddings.

        Returns
        -------
        int or None
            The dimension of the sentence embeddings, or None if it cannot be determined.
        """
        # Use getattr to safely access the out_features attribute of the pooler's dense layer
        return getattr(self.bert.pooler.dense, "out_features", None)

    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids=None):
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

    @staticmethod
    def batch_to_device(batch, device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        return batch

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: Optional[str] = None,
            normalize_embeddings: bool = False,
            max_seq_length: int = None,
    ):
        """
        Returns the embeddings for a batch of sentences.

        :param sentences: str/list, Input sentences
        :param batch_size: int, Batch size
        :param show_progress_bar: bool, Whether to show a progress bar for the sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which device to use for the computation
        :param normalize_embeddings: If true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :param max_seq_length: Override value for max_seq_length
        """
        self.bert.eval()
        if device is None:
            device = self.device
        self.bert.to(device)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if convert_to_tensor:
            convert_to_numpy = False
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            # Compute sentences embeddings
            with torch.no_grad():
                features = self.tokenizer(
                    sentences_batch, max_length=max_seq_length,
                    padding=True, truncation=True, return_tensors='pt'
                )
                features = self.batch_to_device(features, device)
                embeddings = self.get_sentence_embeddings(**features)
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                if convert_to_numpy:
                    embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
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

    @torch.inference_mode()
    def evaluate(self, eval_dataset, output_dir: str = None, batch_size: int = 16):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        results = {}

        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.bert.to(self.device)
        self.bert.eval()

        batch_labels = []
        batch_preds = []
        for batch in tqdm(eval_dataloader, disable=False, desc="Running Evaluation"):
            source, target, labels = batch
            labels = labels.to(self.device)
            batch_labels.extend(labels.cpu().numpy())
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(self.device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(self.device)
            source_token_type_ids = source.get('token_type_ids', None)
            if source_token_type_ids is not None:
                source_token_type_ids = source_token_type_ids.squeeze(1).to(self.device)

            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(self.device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(self.device)
            target_token_type_ids = target.get('token_type_ids', None)
            if target_token_type_ids is not None:
                target_token_type_ids = target_token_type_ids.squeeze(1).to(self.device)

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
        logger.debug(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi processes to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu'] * 4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=SentenceModel._encode_multi_process_worker,
                args=(cuda_id, self, input_queue, output_queue),
                daemon=True
            )
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()

    def encode_multi_process(
            self,
            sentences: List[str],
            pool: Dict[str, object],
            batch_size: int = 32,
            normalize_embeddings: bool = False,
            chunk_size: int = None
    ):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences
        :param pool: A pool of workers started with start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param normalize_embeddings: bool, Whether to normalize embeddings before returning them
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it is a sensible size.
        :return: Numpy matrix with all embeddings
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk, normalize_embeddings])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, normalize_embeddings])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi processes setup
        """
        while True:
            try:
                id, batch_size, sentences, normalize_embeddings = input_queue.get()
                embeddings = model.encode(
                    sentences,
                    device=target_device,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                )
                results_queue.put([id, embeddings])
            except queue.Empty:
                break
