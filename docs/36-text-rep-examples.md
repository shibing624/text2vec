---
title: Text Representation Examples
permalink: /docs/text-rep-examples/
excerpt: "Text Representation Examples"
last_modified_at: 2020/07/26 23:16:38
toc: true
---

### Minimal example for generating word embeddings
Generate a list of contextual word embeddings for every sentence in a list
```python
from simpletransformers.language_representation import RepresentationModel
        
sentences = ["Example sentence 1", "Example sentence 2"]
model = RepresentationModel(
        model_type="bert",
        model_name="bert-base-uncased",
        use_cuda=False
    )
word_vectors = model.encode_sentences(sentences, combine_strategy=None)
assert word_vectors.shape === (2, 5, 768) # token vector for every token in each sentence, bert based models add 2 tokens per sentence by default([CLS] & [SEP])
```
       
### Minimal example for generating sentence embeddings 
Same code as for generating word embeddings, the only difference is that we pass `combine_strategy="mean"` parameter
```python
from simpletransformers.language_representation import RepresentationModel
sentences = ["Example sentence 1", "Example sentence 2"]
model = RepresentationModel(
        model_type="bert",
        model_name="bert-base-uncased",
        use_cuda=False
    )
word_vectors = model.encode_sentences(sentences, combine_strategy="mean")
assert word_vectors.shape === (2, 768) # one sentence embedding per sentence
```
