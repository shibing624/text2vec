# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install jina
"""
from jina import Flow
from docarray import Document, DocumentArray

port = 50001
f = Flow(port=port).add(
    uses='jinahub://Text2vecEncoder',
    uses_with={'model_name': 'shibing624/text2vec-base-chinese'}
)

with f:
    # test example
    r = f.post('/', inputs=DocumentArray([Document(text='如何更换花呗绑定银行卡'), Document(text='花呗更改绑定银行卡')]))
    print(r.embeddings)
    # backend server forever
    f.block()
