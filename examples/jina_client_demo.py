# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import time

sys.path.append('..')
from jina import Client
from docarray import Document

port = 50001
c = Client(port=port)


def encode(sentence):
    """Get one sentence embeddings from jina server."""
    r = c.post('/', inputs=[Document(text=sentence)])
    return r


def batch_encode(sentences):
    """Get sentences' embeddings from jina server."""

    def gen_docs(sent_list):
        for s in sent_list:
            if isinstance(s, str):
                yield Document(text=s)

    r = c.post('/', inputs=gen_docs(sentences), request_size=256)
    return r


if __name__ == '__main__':
    sents = [
        '如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡',
    ]
    print("data:", sents)
    for s in sents:
        r = encode(s)
        print(r.embeddings)
    print('batch embs:', batch_encode(sents).embeddings)

    num_tokens = sum([len(i) for i in sents])
    # QPS test
    for j in range(9):
        batch_sents = sents * (2 ** j)
        c_num_tokens = num_tokens * (2 ** j)
        start_t = time.time()
        r = batch_encode(batch_sents)
        if j == 0:
            print('batch embs:', r.embeddings)
        print('count size:', len(r))
        time_t = time.time() - start_t
        print('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
              (len(batch_sents), time_t, int(len(batch_sents) / time_t), int(c_num_tokens / time_t)))
