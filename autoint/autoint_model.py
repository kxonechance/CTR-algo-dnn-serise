#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@file: autoint_model.py
@function:
@modify:
@reference:
https://github.com/bubblezhong/DeepCTR/blob/master/AutoInt.ipynb
https://blog.csdn.net/zhong_ddbb/article/details/108759521
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def auto_interacting(idx_layer, emb, hidden_factors=16, num_att_head=2):
    num_factors = emb.shape[-1]
    num_fields = emb.shape[1]

    # store the results of self-attention
    attention_heads = []
    w_q = []
    w_k = []
    w_v = []

    # 1. construct many attention
    for i in range(num_att_head):
        # num_factors * hidden_factors
        w_q.append(tf.get_variable('q_{}_{}'.format(i, idx_layer), shape=[num_factors, hidden_factors], initializer=tf.initializers.glorot_normal()))
        w_k.append(tf.get_variable('k_{}_{}'.format(i, idx_layer), shape=[num_factors, hidden_factors], initializer=tf.initializers.glorot_normal()))
        w_v.append(tf.get_variable('v_{}_{}'.format(i, idx_layer), shape=[num_factors, hidden_factors], initializer=tf.initializers.glorot_normal()))

    for i in range(num_att_head):
        # batch * num_fields * hidden_factors
        emb_q = tf.reshape(tf.matmul(tf.reshape(emb, shape=[-1, num_factors]), w_q[i]), shape=[-1, num_fields, hidden_factors])
        emb_k = tf.reshape(tf.matmul(tf.reshape(emb, shape=[-1, num_factors]), w_k[i]), shape=[-1, num_fields, hidden_factors])
        emb_v = tf.reshape(tf.matmul(tf.reshape(emb, shape=[-1, num_factors]), w_v[i]), shape=[-1, num_fields, hidden_factors])

        # calculate the attention
        # batch * num_fields * num_fields
        energy = tf.matmul(emb_q, tf.transpose(emb_k, perm=[0, 2, 1]))
        # batch * num_fields * num_fields
        attention = tf.nn.softmax(energy)

        # batch * num_fields * hidden_factors
        attention_output = tf.matmul(attention, emb_v)
        attention_heads.append(attention_output)

    # 2. concat multi head
    # batch * num_fields * (hidden_factors * num_att_head)
    multi_attention_output = tf.concat(attention_heads, axis=2)

    # 3. ResNet
    # num_factors * (hidden_factors*num_att_head)
    w_res = tf.get_variable('w_res_{}'.format(idx_layer), shape=[num_factors, hidden_factors*num_att_head], initializer=tf.initializers.glorot_normal())
    # batch * num_fields * (hidden_factors * num_att_head)
    output = multi_attention_output + tf.reshape(tf.matmul(tf.reshape(emb, [-1, num_factors]), w_res), shape=[-1, num_fields, hidden_factors*num_att_head])
    # batch * num_fields * (hidden_factors * num_att_head)
    output = tf.nn.relu(output)
    return output


def model_fn(features, labels, mode, params):
    """Build Automatic Feature Interaction Learning via Self-Attentive Neural Networks(AutoInt) model with binary_classification_head."""

    num_dense_features = params['num_dense_features']
    num_sparse_features = params['num_sparse_features']
    num_dense_fields = params['num_dense_fields']
    num_sparse_fields = params['num_sparse_fields']

    learning_rate = params.get('learning_rate', 0.001)

    num_factors = params.get('num_factors', 16)

    hidden_factors = params.get('hidden_factors', 32)
    num_att_heads = params.get('num_att_heads', 2)
    num_att_layers = params.get('num_att_layers', 2)

    dense_indices = tf.reshape(features['dense_indices'], shape=[-1, num_dense_fields])
    dense_values = tf.reshape(features['dense_values'], shape=[-1, num_dense_fields])

    sparse_indices = tf.reshape(features['sparse_indices'], shape=[-1, num_sparse_fields])
    sparse_values = tf.reshape(features['sparse_values'], shape=[-1, num_sparse_fields])

    num_features = num_dense_features + num_sparse_features
    num_fields = num_dense_fields + num_sparse_fields

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, shape=[-1, 1])

    with tf.variable_scope('emb_part'):
        # num_features * num_factors
        w = tf.get_variable('w', shape=[num_features, num_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        # batch * num_fields * num_factors
        embeddings = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * num_factors
        embeddings = tf.multiply(embeddings, tf.reshape(values, shape=(-1, num_fields, 1)))

    with tf.variable_scope('auto_interactive_part'):
        x = embeddings
        for i in range(num_att_layers):
            x = auto_interacting(i, x, hidden_factors, num_att_heads)
        autoint_layer = tf.layers.flatten(x)

    autoint_layer = tf.layers.flatten(embeddings)

    # batch * 1
    with tf.variable_scope('output'):
        logits = tf.layers.dense(autoint_layer, 1)

    my_head = tf.contrib.estimator.binary_classification_head()
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        logits=logits
    )






if __name__ == "__main__":
    pass