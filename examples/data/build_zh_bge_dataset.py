# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: build zh dataset for bge model
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, cast

from datasets import DatasetDict, load_dataset


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


def load_nli():
    dataset = load_dataset('shibing624/nli_zh', 'STS-B')['train']

    def _process(example):
        return {
            'label': 1 if example['label'] > 2.5 else 0,
        }

    dataset = dataset.map(_process)
    dataset = dataset.filter(lambda x: x['label'] == 1)
    corpus = dataset['sentence1'] + dataset['sentence2']

    def add_negatives(example):
        selected_negs = random.sample(corpus, 10)
        if example['sentence1'] in selected_negs:
            selected_negs.remove(example['sentence1'])
        if example['sentence2'] in selected_negs:
            selected_negs.remove(example['sentence2'])
        return {'query': example['sentence1'], 'pos': [example['sentence2']], 'neg': selected_negs}

    dataset_dict = dataset.map(add_negatives, remove_columns=['label', 'sentence1', 'sentence2'])

    dataset_dict = cast(DatasetDict, dataset_dict)
    print(f"processed dataset: {dataset_dict}")
    return dataset_dict


nli_description = DatasetDescription(
    name='nli_zh',
    is_symmetric=True,
    domains=['金融'],
    instruction_type='相似',
)

ALL_DATASETS = [
    NliDataset(load_nli, nli_description),  # 500K
]


def main(output_dir='nli-zh-bge'):
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
