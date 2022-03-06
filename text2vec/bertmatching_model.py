# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Create BERT model for text matching task
"""

import math
import os

import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from text2vec.bertmatching_dataset import BertMatchingTestDataset, BertMatchingTrainDataset, \
    load_test_data, load_train_data, HFBertMatchingTrainDataset, HFBertMatchingTestDataset
from text2vec.sentence_model import device
from text2vec.utils.stats_util import compute_spearmanr, compute_pearsonr
from text2vec.utils.stats_util import set_seed


class BertMatchModule(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = "bert-base-chinese",
            num_classes: int = 2,
    ):
        """
        Initializes a SentenceBert Model.

        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_classes: Number of classes for classification.
        """
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_classes)
        self.bert.to(device)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        """
        Computes the loss for a batch of examples.
        """
        outputs = self.bert(input_ids, token_type_ids, attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        probs = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probs


class BertMatchModel:
    def __init__(
            self,
            model_name_or_path: str = "bert-base-chinese",
            max_seq_length: int = 128,
            num_classes: int = 2,
            encoder_type=None,
    ):
        """
        Initializes the base sentence model.

        :param model_name_or_path: The name of the model to load from the huggingface models library.
        :param max_seq_length: The maximum sequence length.
        :param num_classes: The number of classes.
        :param encoder_type: The encoder type.

        bert model: https://huggingface.co/transformers/model_doc/bert.html?highlight=bert#transformers.BertModel.forward
        BERT return: <last_hidden_state>, <pooler_output> [hidden_states, attentions]
        Note that: in doc, it says <last_hidden_state> is better semantic summery than <pooler_output>.
        thus, we use <last_hidden_state>.
        """
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.model = BertMatchModule(model_name_or_path, num_classes)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.results = {}  # Save training process evaluation result

    def __str__(self):
        return f"<BertMatchModel: {self.model_name_or_path}, max_seq_length: {self.max_seq_length}>"

    def train_model(
            self,
            train_file: str = None,
            output_dir: str = None,
            eval_file: str = None,
            verbose: bool = True,
            batch_size: int = 32,
            num_epochs: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            warmup_ratio: float = 0.1,
            lr: float = 2e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1,
            use_hf_dataset: bool = False,
            hf_dataset_name: str = "STS-B",
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
        Returns:
            global_step: Number of global steps trained
            training_details: Full training progress scores
        """
        if use_hf_dataset and hf_dataset_name:
            logger.info(
                f"Train_file will be ignored when use_hf_dataset is True, load HF dataset: {hf_dataset_name}")
            train_dataset = HFBertMatchingTrainDataset(self.tokenizer, hf_dataset_name, max_len=self.max_seq_length)
            eval_dataset = HFBertMatchingTestDataset(self.tokenizer, hf_dataset_name, max_len=self.max_seq_length)
        elif train_file is not None:
            logger.info(
                f"Hf_dataset_name: {hf_dataset_name} will be ignored when use_hf_dataset is False, load train_file: {train_file}")
            train_dataset = BertMatchingTrainDataset(self.tokenizer, load_train_data(train_file), self.max_seq_length)
            eval_dataset = BertMatchingTestDataset(self.tokenizer, load_test_data(eval_file), self.max_seq_length)
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
            max_steps=max_steps
        )
        logger.info(f" Training model done. Saved to {output_dir}.")

        return global_step, training_details

    def train(
            self,
            train_dataset: Dataset,
            output_dir: str,
            eval_dataset: Dataset = None,
            verbose: bool = True,
            batch_size: int = 8,
            num_epochs: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            warmup_ratio: float = 0.1,
            lr: float = 2e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Use pytorch device: {}".format(device))
        self.model.bert.to(device)
        set_seed(seed)

        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        total_steps = len(train_dataloader) * num_epochs
        param_optimizer = list(self.model.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(total_steps * warmup_ratio)  # by default 10% of _train data for warm-up
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num steps = {total_steps}")
        logger.info(f"  Warmup-steps: {warmup_steps}")

        logger.info("  Training started")
        global_step = 0
        self.model.bert.zero_grad()
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
            self.model.bert.train()
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
                inputs, labels = batch
                labels = labels.to(device)
                # inputs        [batch, 1, seq_len] -> [batch, seq_len]
                input_ids = inputs.get('input_ids').squeeze(1).to(device)
                attention_mask = inputs.get('attention_mask').squeeze(1).to(device)
                token_type_ids = inputs.get('token_type_ids').squeeze(1).to(device)
                loss, logits, probs = self.model(input_ids, attention_mask, token_type_ids, labels)
                current_loss = loss.item()

                if verbose:
                    batch_iterator.set_description(
                        f"Epoch: {epoch_number + 1}/{num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {current_loss:9.4f}")

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.bert.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1
            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))
            results = self.eval_model(eval_dataset, output_dir_current, verbose=verbose, batch_size=batch_size)
            self.save_model(output_dir_current, model=self.model.bert, results=results)
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
                self.save_model(output_dir, model=self.model.bert, results=results)

            if 0 < max_steps < global_step:
                return global_step, training_progress_scores

        return global_step, training_progress_scores

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
        self.model.bert.to(device)
        self.model.bert.eval()

        batch_labels = []
        batch_preds = []
        for batch in tqdm(eval_dataloader, disable=False, desc="Running Evaluation"):
            inputs, labels = batch
            labels = labels.to(device)
            batch_labels.extend(labels.cpu().numpy())
            # inputs        [batch, 1, seq_len] -> [batch, seq_len]
            input_ids = inputs.get('input_ids').squeeze(1).to(device)
            attention_mask = inputs.get('attention_mask').squeeze(1).to(device)
            token_type_ids = inputs.get('token_type_ids').squeeze(1).to(device)

            with torch.no_grad():
                loss, logits, probs = self.model(input_ids, attention_mask, token_type_ids, labels)
            batch_preds.extend(probs.cpu().numpy())

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

    def predict(self, test_dataset, batch_size: int = 16):
        """
        Predicts on test_dataset.
        """
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        self.model.bert.to(device)
        self.model.bert.eval()

        batch_preds = []
        for batch in tqdm(test_dataloader, disable=False, desc="Running Evaluation"):
            inputs, _ = batch
            # inputs        [batch, 1, seq_len] -> [batch, seq_len]
            input_ids = inputs.get('input_ids').squeeze(1).to(device)
            attention_mask = inputs.get('attention_mask').squeeze(1).to(device)
            token_type_ids = inputs.get('token_type_ids').squeeze(1).to(device)

            with torch.no_grad():
                loss, logits, probs = self.model(input_ids, attention_mask, token_type_ids)
            batch_preds.extend(probs.cpu().numpy())

        return batch_preds

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
