# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Create Bge model for text matching task

code modified from https://github.com/FlagOpen/FlagEmbedding
"""

import math
import os

import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm, trange
from transformers.optimization import get_linear_schedule_with_warmup

from text2vec.bge_dataset import BgeTrainDataset
from text2vec.sentence_model import SentenceModel
from text2vec.text_matching_dataset import TextMatchingTestDataset, load_text_matching_test_data
from text2vec.utils.stats_util import set_seed


class BgeModel(SentenceModel):
    def __init__(
            self,
            model_name_or_path: str = "BAAI/bge-large-zh-noinstruct",
            encoder_type: str = "MEAN",
            max_seq_length: int = 32,
            passage_max_len: int = 128,
            device: str = None,
    ):
        """
        Initializes a Bge Model.

        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            encoder_type: encoder type, set by model name
            max_seq_length: The maximum total input sequence length after tokenization.
            passage_max_len: The maximum total input sequence length after tokenization.
            num_classes: Number of classes for classification.
            device: CPU or GPU
        """
        super().__init__(model_name_or_path, encoder_type, max_seq_length, device)
        self.query_max_len = max_seq_length
        self.passage_max_len = passage_max_len

    def __str__(self):
        return f"<BgeModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}>"

    def train_model(
            self,
            train_file: str = None,
            output_dir: str = None,
            eval_file: str = None,
            verbose: bool = True,
            batch_size: int = 32,
            num_epochs: int = 1,
            weight_decay: float = 0.0,
            seed: int = 42,
            warmup_ratio: float = 0.05,
            lr: float = 1e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1,
            use_hf_dataset: bool = False,
            hf_dataset_name: str = "",
            save_model_every_epoch: bool = True,
            bf16: bool = False,
            data_parallel: bool = False,
            train_group_size: int = 8,
            temperature: float = 1.0,
            normalize_embeddings: bool = False,
    ):
        """
        Trains the model on 'train_file'

        Args:
            train_file: Path to text file containing the text to _train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            eval_file (optional): Path to eval file containing the text to _evaluate the language model on.
            verbose (optional): Print logger or not.
            batch_size (optional): Batch size for training.
            num_epochs (optional): Number of epochs for training.
            weight_decay (optional): Weight decay for optimization.
            seed (optional): Seed for initialization.
            warmup_ratio (optional): Warmup ratio for learning rate.
            lr (optional): Learning rate.
            eps (optional): Adam epsilon.
            gradient_accumulation_steps (optional): Number of updates steps to accumulate before performing a backward/update pass.
            max_grad_norm (optional): Max gradient norm.
            max_steps (optional): If > 0: set total number of training steps to perform. Override num_epochs.
            use_hf_dataset (optional): Whether to use the HuggingFace datasets for training.
            hf_dataset_name (optional): Name of the dataset to use for the HuggingFace datasets.
            save_model_every_epoch (optional): Save model checkpoint every epoch.
            bf16 (optional): Use bfloat16 amp training.
            data_parallel (optional): Use multi-gpu data parallel training.
            train_group_size (optional): Group size for training.
            temperature (optional): Temperature for softmax.
            normalize_embeddings (optional): Normalize embeddings or not.
        Returns:
            global_step: Number of global steps trained
            training_details: Full training progress scores
        """
        if use_hf_dataset and hf_dataset_name:
            logger.info(
                f"Train_file will be ignored when use_hf_dataset is True, load HF dataset: {hf_dataset_name}")
            train_dataset = BgeTrainDataset(self.tokenizer, hf_dataset_name, self.query_max_len, self.passage_max_len,
                                            train_group_size)
            eval_dataset = None
        elif train_file is not None:
            logger.info(f"Load train_file: {train_file}")
            train_dataset = BgeTrainDataset(self.tokenizer, train_file, self.query_max_len, self.passage_max_len,
                                            train_group_size)
            eval_dataset = TextMatchingTestDataset(self.tokenizer, load_text_matching_test_data(eval_file),
                                                   self.max_seq_length)
        else:
            raise ValueError("Error, train_file|use_hf_dataset must be specified")

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            eval_dataset=eval_dataset,
            verbose=verbose,
            batch_size=batch_size,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            seed=seed,
            warmup_ratio=warmup_ratio,
            lr=lr,
            eps=eps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            save_model_every_epoch=save_model_every_epoch,
            bf16=bf16,
            data_parallel=data_parallel,
            temperature=temperature,
            normalize_embeddings=normalize_embeddings,
        )
        logger.info(f" Training model done. Saved to {output_dir}.")

        return global_step, training_details

    def calc_loss(self, y_true, y_pred):
        """
        Calc loss with two sentence embeddings, Softmax loss
        """
        loss = nn.CrossEntropyLoss(reduction='mean')(y_pred, y_true)
        return loss

    def calc_similarity(self, q_embs, p_embs):
        """
        Calc similarity with two sentence embeddings
        """
        if len(p_embs.size()) == 2:
            return torch.matmul(q_embs, p_embs.transpose(0, 1))
        return torch.matmul(q_embs, p_embs.transpose(-2, -1))

    @staticmethod
    def flat_list(l):
        return [item for sublist in l for item in sublist]

    def train(
            self,
            train_dataset: Dataset,
            output_dir: str,
            eval_dataset: Dataset = None,
            verbose: bool = True,
            batch_size: int = 8,
            num_epochs: int = 1,
            weight_decay: float = 0.0,
            seed: int = 42,
            warmup_ratio: float = 0.05,
            lr: float = 1e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1,
            save_model_every_epoch: bool = True,
            bf16: bool = False,
            data_parallel: bool = False,
            temperature: float = 1.0,
            normalize_embeddings: bool = False,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Use device: {}".format(self.device))
        self.bert.to(self.device)
        set_seed(seed)
        num_devices = 1
        torch_type = torch.bfloat16 if bf16 else torch.float32

        if data_parallel:
            self.bert = nn.DataParallel(self.bert)
            num_devices = torch.cuda.device_count()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            sampler = DistributedSampler(train_dataset, num_replicas=num_devices, rank=local_rank)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # not shuffle
        total_steps = len(train_dataloader) * num_epochs // num_devices
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(total_steps * warmup_ratio)  # by default 10% of _train data for warm-up
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num steps = {total_steps}")
        logger.info(f"  Warmup-steps: {warmup_steps}")

        logger.info("  Training started")
        global_step = 0
        self.bert.zero_grad()
        epoch_number = 0
        best_eval_metric = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if self.model_name_or_path and os.path.exists(self.model_name_or_path):
            try:
                # set global_step to global_step of last saved checkpoint from model path
                checkpoint_suffix = self.model_name_or_path.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)
                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d" % epochs_trained)
                logger.info("   Continuing training from global step %d" % global_step)
                logger.info("   Will skip the first %d steps in the current epoch" % steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "eval_spearman": [],
            "eval_pearson": [],
        }

        for current_epoch in trange(int(num_epochs), desc="Epoch", disable=False, mininterval=0):
            self.bert.train()
            current_loss = 0
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            batch_iterator = tqdm(train_dataloader,
                                  desc=f"Running Epoch {epoch_number + 1} of {num_epochs}",
                                  disable=False,
                                  mininterval=0)
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                query, passage = batch
                query = self.flat_list(query)
                passage = self.flat_list(passage)
                query = self.tokenizer(
                    query,
                    max_length=self.query_max_len,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                passage = self.tokenizer(
                    passage,
                    max_length=self.passage_max_len,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                query = query.to(self.device)
                passage = passage.to(self.device)

                # get sentence embeddings
                with torch.autocast(str(self.device), dtype=torch_type):
                    q_embeddings = self.get_sentence_embeddings(**query)
                    p_embeddings = self.get_sentence_embeddings(**passage)
                    if normalize_embeddings:
                        q_embeddings = torch.nn.functional.normalize(q_embeddings, dim=-1)
                        p_embeddings = torch.nn.functional.normalize(p_embeddings, dim=-1)

                    scores = self.calc_similarity(q_embeddings, p_embeddings)
                    scores = scores / temperature
                    scores = scores.view(q_embeddings.size(0), -1)

                    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                    target = target * (p_embeddings.size(0) // q_embeddings.size(0))
                    loss = self.calc_loss(target, scores)
                current_loss = loss.item()
                if verbose:
                    batch_iterator.set_description(
                        f"Epoch: {epoch_number + 1}/{num_epochs}, "
                        f"Batch:{step}/{len(train_dataloader)//num_devices}, Loss: {current_loss:9.4f}")

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.bert.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1
            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))
            results = self.eval_model(eval_dataset, output_dir_current, verbose=verbose, batch_size=batch_size)
            if save_model_every_epoch:
                self.save_model(output_dir_current, model=self.bert, results=results)
            training_progress_scores["global_step"].append(global_step)
            training_progress_scores["train_loss"].append(current_loss)
            for key in results:
                training_progress_scores[key].append(results[key])
            report = pd.DataFrame(training_progress_scores)
            report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

            eval_spearman = results["eval_spearman"]
            if eval_spearman > best_eval_metric:
                best_eval_metric = eval_spearman
                logger.info(f"Save new best model, best_eval_metric: {best_eval_metric}")
                self.save_model(output_dir, model=self.bert, results=results)

            if 0 < max_steps < global_step:
                return global_step, training_progress_scores

        return global_step, training_progress_scores
