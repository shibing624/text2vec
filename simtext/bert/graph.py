# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""

import json
import tensorflow as tf
import os
from . import modeling
from ..utils import get_logger
logger = get_logger(__name__)


def optimize_graph(layer_indexes=[-2], config_name = '', ckpt_name = '', max_seq_len=128, output_dir=''):
    try:
        # we don't need GPU for optimizing the graph
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

        # allow_soft_placement:自动选择运行设备
        config = tf.ConfigProto(allow_soft_placement=True)
        config_fp = config_name
        init_checkpoint = ckpt_name
        logger.info('model config: %s' % config_fp)

        # 加载bert配置文件
        with tf.gfile.GFile(config_fp, 'r') as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))

        logger.info('build graph...')
        # input placeholders, not sure if they are friendly to XLA
        input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
        input_type_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_type_ids')

        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

        with jit_scope():
            input_tensors = [input_ids, input_mask, input_type_ids]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False)

            # 获取所有要训练的变量
            tvars = tf.trainable_variables()

            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            # 共享卷积核
            with tf.variable_scope("pooling"):
                # 如果只有一层，就只取对应那一层的weight
                if len(layer_indexes) == 1:
                    encoder_layer = model.all_encoder_layers[layer_indexes[0]]
                else:
                    # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
                    all_layers = [model.all_encoder_layers[l] for l in layer_indexes]
                    encoder_layer = tf.concat(all_layers, -1)

                input_mask = tf.cast(input_mask, tf.float32)

            # 以下代码是句向量的生成方法，可以理解为做了一个卷积的操作，但是没有把结果相加, 卷积核是input_mask
            pooled = masked_reduce_mean(encoder_layer, input_mask)
            pooled = tf.identity(pooled, 'final_encodes')

            output_tensors = [pooled]
            bert_graph = tf.get_default_graph().as_graph_def()

        with tf.Session(config=config) as sess:
            logger.info('load parameters from checkpoint...')
            sess.run(tf.global_variables_initializer())
            logger.info('freeze...')
            bert_graph = tf.graph_util.convert_variables_to_constants(sess, bert_graph, [n.name[:-2] for n in output_tensors])
            dtypes = [n.dtype for n in input_tensors]
            logger.info('optimize...')
            bert_graph = optimize_for_inference(
                bert_graph,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        graph_file = os.path.join(output_dir, 'graph.txt')
        logger.info('write graph to file: %s' % graph_file)
        with tf.gfile.GFile(graph_file, 'wb') as f:
            f.write(bert_graph.SerializeToString())
        return graph_file
    except Exception as e:
        logger.error('fail to optimize the graph!')
        logger.error(e)
