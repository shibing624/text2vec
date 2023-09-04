# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

code from https://github.com/FlagOpen/FlagEmbedding
"""

import argparse
import json
import random
import sys

import faiss
from tqdm import tqdm

sys.path.append('../..')
from text2vec import SentenceModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-large-zh-noinstruct", type=str)
    parser.add_argument('--input_file', default='nli-zh-bge/nli_zh-train.jsonl', type=str)
    parser.add_argument('--candidate_pool', default='STS-B/STS-B.train.data', type=str)
    parser.add_argument('--output_file', default='bge_finetune_data.jsonl', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--range_for_sampling', default='2-20', type=str, help="range to sample negatives")
    parser.add_argument('--use_gpu_for_searching', action='store_true', help='use faiss-gpu')
    parser.add_argument('--negative_number', default=10, help='use faiss-gpu')
    return parser.parse_args()


def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    if use_gpu:
        print('use faiss-gpu')
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(
        index,
        query,
        topk: int = 200,
        batch_size: int = 64
):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < batch_size):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(batch_query, k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool, 'r', encoding='utf-8'):
        parts = line.strip().split('\t')
        txt1 = parts[0].strip()
        txt2 = parts[1].strip()
        corpus.append(txt1)
        corpus.append(txt2)
    return corpus


def find_knn_neg(
        model,
        input_file,
        candidate_pool,
        output_file,
        sample_range,
        negative_number,
        use_gpu,
        batch_size
):
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file, 'r', encoding='utf-8'):
        line = json.loads(line.strip())
        train_data.append(line)
        corpus.extend(line['neg'])
        queries.append(line['query'])

    if candidate_pool is not None:
        corpus = get_corpus(candidate_pool)
    corpus = list(set(corpus))

    print(f'inference embedding for corpus (number={len(corpus)})--------------')
    p_vecs = model.encode(corpus, batch_size=batch_size, normalize_embeddings=True)
    print(f'inference embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode(queries, batch_size=batch_size, normalize_embeddings=True)

    print('create index and search------------------')
    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1], batch_size=batch_size)
    assert len(all_inxs) == len(train_data)

    for i, data in enumerate(train_data):
        query = data['query']
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1:
                break
            if corpus[inx] not in data['pos'] and corpus[inx] != query:
                filtered_inx.append(inx)

        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        data['neg'] = [corpus[inx] for inx in filtered_inx]

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in train_data:
            if len(data['neg']) < negative_number:
                data['neg'].extend(random.sample(corpus, negative_number - len(data['neg'])))
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = get_args()
    print(args)
    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    model = SentenceModel(args.model_name_or_path)

    find_knn_neg(
        model,
        input_file=args.input_file,
        candidate_pool=args.candidate_pool,
        output_file=args.output_file,
        sample_range=sample_range,
        negative_number=args.negative_number,
        use_gpu=args.use_gpu_for_searching,
        batch_size=args.batch_size
    )
