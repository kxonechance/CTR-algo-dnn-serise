#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@file: fibinet_model.py
@function:
@modify:  reference: https://github.com/qiaoguan/deep-ctr-prediction/blob/master/Fibinet/fibinet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build FiBiNet model with binary_classification_head."""

    num_dense_features = params['num_dense_features']
    num_sparse_features = params['num_sparse_features']
    num_dense_fields = params['num_dense_fields']
    num_sparse_fields = params['num_sparse_fields']

    learning_rate = params.get('learning_rate', 0.001)

    num_factors = params.get('num_factors', 4)
    reduce_ratio = params.get('num_factors', 4.0)
    deep_layers = params.get('deep_layers', [100, 100])

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

    with tf.variable_scope('SENET_part'):
        reduce_factors = max(1, num_fields // reduce_ratio)
        # num_fields * reduce_factors
        w1 = tf.get_variable('w1', shape=[num_fields, reduce_factors], initializer=tf.initializers.glorot_normal())
        # reduce_factors * num_fields
        w2 = tf.get_variable('w2', shape=[reduce_factors, num_fields], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        r = tf.reduce_sum(embeddings, axis=2, keepdims=False)
        # batch * reduce_factors
        r1 = tf.matmul(r, w1)
        # batch * num_fields
        weight = tf.matmul(r1, w2)
        # batch * num_fields * num_factors
        w_embeddings = tf.multiply(embeddings, tf.reshape(weight, shape=(-1, num_fields, 1)))

    with tf.variable_scope('BiLinear_interaction_part'):
        emb_list = [embeddings, w_embeddings]
        bilinear_res = []
        for k, embeddings in enumerate(emb_list):
            element_wise_product_list = []
            for i in range(num_fields):
                for j in range(i+1, num_fields):
                    w = tf.get_variable('w_{}_{}_{}'.format(k, i, j), shape=[num_factors, num_factors], initializer=tf.initializers.glorot_normal())
                    element_wise_product_list.append(tf.multiply(tf.matmul(embeddings[:, i, :], w), embeddings[:, j, :]))
            # ((num_fields)*(num_fields-1)/2) * batch * num_factors
            element_wise_product = tf.stack(element_wise_product_list)
            # batch*((num_fields)*(num_fields-1)/2)*num_factors
            element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2])
            element_wise_product = tf.reshape(element_wise_product, shape=[-1, int(num_fields*(num_fields-1)/2)*num_factors])
            bilinear_res.append(element_wise_product)

    with tf.variable_scope('combination_layer'):
        # batch * num_fields*(num_fields-1)*num_factors
        comb_part = tf.concat(bilinear_res, axis=1)

    with tf.variable_scope('deep_layer'):
        for i in range(len(deep_layers)):
            if i == 0:
                dense = tf.layers.dense(comb_part, deep_layers[i], activation=tf.nn.relu)
                dense = tf.layers.batch_normalization(dense, training=mode == tf.estimator.ModeKeys.TRAIN)
                dense = tf.layers.dropout(dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
            else:
                dense = tf.layers.dense(dense, deep_layers[i], activation=tf.nn.relu)
                dense = tf.layers.batch_normalization(dense, training=mode == tf.estimator.ModeKeys.TRAIN)
                dense = tf.layers.dropout(dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # batch * 1
    with tf.variable_scope('output'):
        logits = tf.layers.dense(dense, 1)

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