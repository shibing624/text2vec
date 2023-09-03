# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: build zh nli dataset

part of this code modified from https://github.com/wangyuxinwhy/uniem/blob/main/scripts/process_zh_datasets.py
"""
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


@dataclass
class DatasetDescription:
    name: str
    is_symmetric: bool
    domains: List[str]
    instruction_type: str


@dataclass
class NliDataset:
    load_fn: Callable[[], DatasetDict]
    description: DatasetDescription


def add_label(examples):
    return {'label': [1 for _ in examples['text1']]}


def load_cmrc2018():
    dataset_dict = load_dataset('cmrc2018')
    print(dataset_dict)
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'question': 'text2', 'context': 'text1'})
    dataset_dict = dataset_dict.map(add_label, batched=True, remove_columns=['id', 'answers'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


cmrc2018_description = DatasetDescription(
    name='cmrc2018',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_belle():
    dataset_dict = load_dataset('BelleGroup/train_0.5M_CN')
    print(dataset_dict)
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'instruction': 'text1', 'output': 'text2'})
    dataset_dict = dataset_dict.map(add_label, batched=True, remove_columns=['input'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


belle_description = DatasetDescription(
    name='belle',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='',
)


def load_firefly():
    dataset_dict = load_dataset('YeungNLP/firefly-train-1.1M')
    if isinstance(dataset_dict, Dataset):
        dataset_dict = DatasetDict({'train': dataset_dict})
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.filter(lambda x: x['kind'] in ['Summary', 'Couplet', 'MusicComment',
                                                               'TextCorrection', 'ProductDesc', 'OpenQA',
                                                               'ClassicalChinese', 'Dictionary', 'Composition',
                                                               'StoryGeneration',
                                                               'MRC', 'JinYongGeneration', 'LyricGeneration',
                                                               'Translation'])
    dataset_dict = dataset_dict.rename_columns({'input': 'text1', 'target': 'text2'})
    dataset_dict = dataset_dict.map(add_label, batched=True, remove_columns=['kind'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


firefly_description = DatasetDescription(
    name='firefly',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='',
)


def load_alpaca_gpt4():
    dataset_dict = load_dataset('shibing624/alpaca-zh')
    if isinstance(dataset_dict, Dataset):
        dataset_dict = DatasetDict({'train': dataset_dict})
    dataset_dict = cast(DatasetDict, dataset_dict)

    def concat_instruction_and_input(batch):
        return {
            'text1': [f'{instruction} {input}' for instruction, input in zip(batch['instruction'], batch['input'])],
        }

    dataset_dict = dataset_dict.map(concat_instruction_and_input, batched=True)
    dataset_dict = dataset_dict.rename_columns({'text1': 'text1', 'output': 'text2'})
    dataset_dict = dataset_dict.map(add_label, batched=True, remove_columns=['instruction', 'input'])
    return dataset_dict


alpaca_gpt4_description = DatasetDescription(
    name='alpaca_gpt4',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='',
)


def load_zhihu_kol():
    dataset_dict = load_dataset('wangrui6/Zhihu-KOL')
    print(dataset_dict)
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'INSTRUCTION': 'text1', 'RESPONSE': 'text2'})
    dataset_dict = dataset_dict.map(add_label, batched=True, remove_columns=['SOURCE', 'METADATA'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


zhihu_kol_description = DatasetDescription(
    name='zhihu_kol',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_wiki_atomic_edits():
    letters_and_digits = set(string.ascii_letters + string.digits)

    dataset1 = load_dataset('wiki_atomic_edits', 'chinese_insertions')['train']  # type: ignore
    dataset2 = load_dataset('wiki_atomic_edits', 'chinese_deletions')['train']  # type: ignore
    dataset = concatenate_datasets([dataset1, dataset2])  # type: ignore

    def concat_words(words):
        text = ''
        for word in words:
            if word[0] in letters_and_digits or word[-1] in letters_and_digits:
                word = ' ' + word + ' '
            text += word
        text = text.strip()
        text = text.replace('  ', ' ')
        return text

    def _process(example):
        return {
            'text1': concat_words(example['base_sentence'].split(' ')),
            'text2': concat_words(example['edited_sentence'].split(' ')),
        }

    dataset = dataset.map(_process)
    dataset = dataset.map(add_label, batched=True, remove_columns=['id', 'phrase', 'base_sentence', 'edited_sentence'])
    print(f"processed dataset: {dataset}")
    return dataset


wiki_atomic_edis_description = DatasetDescription(
    name='wiki_atomic_edits',
    is_symmetric=True,
    domains=['百科'],
    instruction_type='相似',
)


def load_chatmed_consult():
    dataset_dict = load_dataset('michaelwzhu/ChatMed_Consult_Dataset')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'query': 'text1', 'response': 'text2'})
    dataset_dict = dataset_dict.map(add_label, batched=True)
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


chatmed_consult_description = DatasetDescription(
    name='chatmed_consult',
    is_symmetric=False,
    domains=['医药'],
    instruction_type='问答',
)


def load_amazon_reviews():
    dataset_dict = load_dataset('amazon_reviews_multi', 'zh')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'review_title': 'text2', 'review_body': 'text1'})
    dataset_dict = dataset_dict.map(add_label, batched=True,
                                    remove_columns=['review_id', 'product_id', 'reviewer_id', 'stars', 'language',
                                                    'product_category'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


amazon_reviews_description = DatasetDescription(
    name='amazon_reviews',
    is_symmetric=False,
    domains=['电商'],
    instruction_type='摘要',
)


def load_xlsum():
    dataset = load_dataset('csebuetnlp/xlsum', 'chinese_simplified')
    dataset1 = dataset.select_columns(['title', 'summary'])
    dataset2 = dataset.select_columns(['title', 'text'])
    dataset1 = dataset1.rename_columns({'title': 'text1', 'summary': 'text2'})
    dataset2 = dataset2.rename_columns({'title': 'text1', 'text': 'text2'})
    all_datasets = list(dataset1.values()) + list(dataset2.values())  # type: ignore
    dataset = concatenate_datasets(all_datasets)
    dataset = dataset.map(add_label, batched=True)
    print(f"processed dataset: {dataset}")
    return dataset


xlsum_description = DatasetDescription(
    name='xlsum',
    is_symmetric=False,
    domains=['新闻'],
    instruction_type='摘要',
)


def load_mlqa():
    dataset_dict = load_dataset('mlqa', 'mlqa-translate-train.zh')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'question': 'text2', 'context': 'text1'})
    dataset = dataset_dict.map(add_label, batched=True, remove_columns=['id', 'answers'])
    print(f"processed dataset: {dataset}")
    return dataset


mlqa_description = DatasetDescription(
    name='mlqa',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_ocnli():
    dataset_dict = load_dataset('dirtycomputer/OCNLI')
    dataset_dict = cast(DatasetDict, dataset_dict)

    def _process(example):
        return {
            'label': 1 if example['label'] == "entailment" else 0,
        }

    dataset_dict = dataset_dict.map(_process,
                                    remove_columns=['id', 'level', 'label0', 'label1', 'label2', 'label3', 'label4',
                                                    'genre', 'prem_id'])
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text1', 'sentence2': 'text2'})
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


ocnli_description = DatasetDescription(
    name='ocnli',
    is_symmetric=True,
    domains=['口语'],
    instruction_type='推理',
)


def load_nli():
    dataset1 = load_dataset('shibing624/nli_zh', 'BQ')['train']
    dataset2 = load_dataset('shibing624/nli_zh', 'LCQMC')['train']
    dataset3 = load_dataset('shibing624/nli_zh', 'PAWSX')['train']
    dataset4 = load_dataset('shibing624/nli_zh', 'ATEC')['train']
    dataset5 = load_dataset('shibing624/nli_zh', 'STS-B')['train']

    def _process(example):
        return {
            'label': 1 if example['label'] > 2.5 else 0,
        }

    dataset5 = dataset5.map(_process)
    dataset_dict = concatenate_datasets([dataset1, dataset2, dataset3, dataset4, dataset5])  # type: ignore

    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text1', 'sentence2': 'text2'})
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


nli_description = DatasetDescription(
    name='nli_zh',
    is_symmetric=True,
    domains=['金融'],
    instruction_type='相似',
)


def load_webqa():
    dataset_dict = load_dataset('suolyer/webqa')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'input': 'text1', 'output': 'text2'})
    dataset_dict = dataset_dict.map(add_label, batched=True, remove_columns=['id'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


webqa_description = DatasetDescription(
    name='webqa',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_csl():
    dataset_dict = load_dataset('neuclir/csl')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'title': 'text1', 'abstract': 'text2'})
    dataset_dict = dataset_dict.map(add_label, batched=True,
                                    remove_columns=['doc_id', 'keywords', 'category', 'category_eng', 'discipline',
                                                    'discipline_eng'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


csl_description = DatasetDescription(
    name='csl',
    is_symmetric=False,
    domains=['学术'],
    instruction_type='摘要',
)


def load_dureader_robust():
    dataset_dict = load_dataset('PaddlePaddle/dureader_robust')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'question': 'text2', 'context': 'text1'})
    dataset_dict = dataset_dict.map(add_label, batched=True, remove_columns=['id', 'title', 'answers'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


dureader_robust_description = DatasetDescription(
    name='dureader_robust',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_cblue_chip_sts():
    dataset_dict = load_dataset('dirtycomputer/CBLUE-CHIP-STS')
    dataset_dict = cast(DatasetDict, dataset_dict)

    def _process(example):
        return {
            'label': 1 if example['label'] == "1" else 0,
        }

    dataset_dict = dataset_dict.map(_process, remove_columns=['id', 'category'])
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


cblue_chip_sts_description = DatasetDescription(
    name='cblue_chip_sts',
    is_symmetric=True,
    domains=['医疗'],
    instruction_type='相似',
)


def load_snli_zh():
    dataset_dict = load_dataset('shibing624/snli-zh')
    dataset_dict = cast(DatasetDict, dataset_dict)

    def _process(example):
        return {
            'label': 1 if example['label'] == 0 else 0,
        }

    dataset_dict = dataset_dict.rename_columns({'premise': 'text1', 'hypothesis': 'text2'})
    dataset_dict = dataset_dict.map(_process)
    dataset_dict = dataset_dict.filter(lambda x: x['text1'] and x['text2'] and len(x['text1']) > 1 and len(x['text2']) > 1)
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


snli_zh_description = DatasetDescription(
    name='snli_zh',
    is_symmetric=True,
    domains=['口语'],
    instruction_type='推理',
)



def load_simclue():
    dataset_dict = load_dataset('alexwww94/SimCLUE', 'train_pair')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text1', 'sentence2': 'text2'})
    dataset_dict = dataset_dict.filter(lambda x: x['text1'] and x['text2'] and len(x['text1']) > 1 and len(x['text2']) > 1)
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


simclue_description = DatasetDescription(
    name='simclue',
    is_symmetric=True,
    domains=['口语'],
    instruction_type='推理',
)

ALL_DATASETS = [
    NliDataset(load_cmrc2018, cmrc2018_description),  # 10K
    NliDataset(load_belle, belle_description),  # 0.5M
    NliDataset(load_firefly, firefly_description),  # 1.6M
    NliDataset(load_alpaca_gpt4, alpaca_gpt4_description),  # 5K
    NliDataset(load_zhihu_kol, zhihu_kol_description),  # 1.1M
    NliDataset(load_wiki_atomic_edits, wiki_atomic_edis_description),  # 1.1M
    NliDataset(load_chatmed_consult, chatmed_consult_description),  # 549K
    NliDataset(load_amazon_reviews, amazon_reviews_description),  # 210K
    NliDataset(load_xlsum, xlsum_description),  # 46.7K
    NliDataset(load_mlqa, mlqa_description),  # 85K
    NliDataset(load_ocnli, ocnli_description),  # 50K
    NliDataset(load_nli, nli_description),  # 500K
    NliDataset(load_webqa, webqa_description),  # 32K
    NliDataset(load_csl, csl_description),  # 396K
    NliDataset(load_dureader_robust, dureader_robust_description),  # 14K
    NliDataset(load_cblue_chip_sts, cblue_chip_sts_description),  # 16K
    NliDataset(load_snli_zh, snli_zh_description),  # 420K
    NliDataset(load_simclue, simclue_description),  # 2.6M
]


def main(output_dir='nli-zh-all'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for ds in ALL_DATASETS:
        f = output_dir / f'{ds.description.name}-train.jsonl'
        if f.exists():
            print(f'{f} exists, skip')
            continue

        dataset = ds.load_fn()
        if hasattr(dataset, 'items'):
            for split, dataset in dataset.items():
                if split != 'train':
                    continue

        dataset.to_json(f"{f}", lines=True, force_ascii=False)


if __name__ == '__main__':
    main()
