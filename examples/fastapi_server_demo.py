# -*- coding: utf-8 -*-


import argparse
import uvicorn
import sys
import os
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
import torch
from loguru import logger
from typing import List
from pydantic import BaseModel, Field
import numpy as np


sys.path.append('..')
from text2vec import SentenceModel

class Item(BaseModel):
    input: str = Field(..., max_length=512)

pwd_path = os.path.abspath(os.path.dirname(__file__))
use_cuda = torch.cuda.is_available()
logger.info(f'use_cuda:{use_cuda}')
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="shibing624/text2vec-base-chinese",
                    help="Model save dir or model name")
args = parser.parse_args()
s_model = SentenceModel(args.model_name_or_path)

def _normalize_embedding_2D(vec: np.ndarray) -> np.ndarray:
  vec = np.ascontiguousarray(vec)
  norm = np.sqrt(vec.dot(vec))
  if norm != 0.0:
    vec /= norm
  return vec

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get('/')
async def index():
    return {"message": "index, docs url: /docs"}


@app.post('/emb')
async def emb(item: Item):
    try:
        embeddings = s_model.encode(item.input)
        embeddings = np.array(embeddings)
        normalized_embeddings = _normalize_embedding_2D(embeddings)
        result_dict = {'emb': normalized_embeddings.tolist()}
        logger.debug(f"Successfully get sentence embeddings, q:{item.input}")
        return result_dict
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)

