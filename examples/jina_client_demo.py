# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import time
import sys

sys.path.append('..')
from jina import Client
from docarray import Document

port = 50001
c = Client(port=port)


def jina_post():
    r = c.post('/', inputs=[Document(text='如何更换花呗绑定银行卡'), Document(text='花呗更改绑定银行卡')])
    return r


def encode(c, sentences):
    def gen_docs(sent_list):
        for s in sent_list:
            if isinstance(s, str):
                yield Document(text=s)

    r = c.post('/', inputs=gen_docs(sentences), request_size=256)
    return r


if __name__ == '__main__':
    data = ['如何更换花呗绑定银行卡',
            '花呗更改绑定银行卡']
    print("data:", data)
    r = jina_post()
    print(r.embeddings)
    print('batch embs:', encode(c, data).embeddings)

    num_tokens = sum([len(i) for i in data])
    # QPS test
    for j in range(9):
        tmp = data * (2 ** j)
        c_num_tokens = num_tokens * (2 ** j)
        start_t = time.time()
        r = encode(c, tmp)
        if j == 0:
            print('batch embs:', r.embeddings)
        print('count size:', len(r))
        time_t = time.time() - start_t
        print('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
              (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))
