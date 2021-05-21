---
title: Text Representation Model
permalink: /docs/text-rep-model/
excerpt: "Text Representation Model"
last_modified_at: 2020/07/26 23:16:38
toc: true
---



## `RepresentationModel`

The `RepresentationModel` class is used for generating (contextual) word or sentence embeddings from a list of text sentences,
You can then feed these vectors to any model or downstream task.

To create a `RepresentationModel`, you must specify a `model_type` and a `model_name`.

- `model_type` should be one of the model types, currently supported: bert, roberta, gpt2
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

    **Note:** For a list of standard pre-trained models, see [here](https://huggingface.co/transformers/pretrained_models.html).
    {: .notice--info}

    **Note:** For a list of community models, see [here](https://huggingface.co/models).
    {: .notice--info}

    You may use any of these models provided the `model_type` is supported.

```python
from simpletransformers.language_representation import RepresentationModel

model = RepresentationModel(
    "roberta", "roberta-base"
)
```

**Note:** For more information on working with Simple Transformers models, please refer to the [General Usage section](/docs/usage/#creating-a-task-specific-model).
{: .notice--info}


### Configuring a `RepresentationModel`


```python
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

model_args = ModelArgs(max_seq_length=156)

model = RepresentationModel(
    "roberta",
    "roberta-base",
    args=model_args,
)
```

**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}


## `Class RepresentationModel`

> *simpletransformers.language_representation.RepresentationModel*{: .function-name}(self, model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs,)

Initializes a RepresentationModel model.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **model_type** *(`str`)* - The type of model to use, currently supported: bert, roberta, gpt2

* **model_name** *(`str`)* - The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

* **args** *(`dict`, optional)* - [Default args](/docs/usage/#configuring-a-simple-transformers-model) will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.

* **use_cuda** *(`bool`, optional)* - Use GPU if available. Setting to False will force model to use CPU only. (See [here](/docs/usage/#to-cuda-or-not-to-cuda))

* **cuda_device** *(`int`, optional)* - Specific GPU that should be used. Will use the first available GPU by default. (See [here](/docs/usage/#selecting-a-cuda-device))

* **kwargs** *(optional)* - For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied. (See [here](/docs/usage/#options-for-downloading-pre-trained-models))
{: .parameter-list}

> Returns
{: .returns}

* `None`
{: .return-list}


**Note:** For configuration options common to all Simple Transformers models, please refer to the [Configuring a Simple Transformers Model section](/docs/usage/#configuring-a-simple-transformers-model).
{: .notice--info}



## Generating contextual word embeddings from text with a `RepresentationModel`

The `encode_sentences()`  method is used to create word embeddings with the model.

```python
sentence_list = ["Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages", "hi", "I am happy"]
word_embeddings = model.encode_sentences(sentence_list, combine_strategy="mean")
```

**Note:** The input **must** be a List even if there is only one sentence.
{: .notice--info}


> *simpletransformers.language_representation.RepresentationModel.encode_sentences*{: .function-name}(text_list, combine_strategy=None, batch_size=32)

Generates list of contextual word or sentence embeddings using the model passed to class constructor.
{: .function-text}

> Parameters
{: .parameter-blockquote}

* **text_list** - list of text sentences.

* **combine_strategy** - strategy for combining word vectors, supported values: None, "mean", "concat".

* **batch_size** - size of batches of sentences feeded to the model.
{: .parameter-list}

> Returns
{: .returns}

* **answer_list** *(`list`)* - list of lists of sentence embeddings(if `combine_strategy=None`) OR list of sentence embeddings(if `combine_strategy!=None`).
{: .return-list}
