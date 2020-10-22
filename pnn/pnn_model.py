#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@file: pnn_model.py
@function:
@modify:  reference: https://www.cnblogs.com/gogoSandy/p/12742417.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build Product Neural Network(PNN) model with binary_classification_head."""

    num_dense_features = params['num_dense_features']
    num_sparse_features = params['num_sparse_features']
    num_dense_fields = params['num_dense_fields']
    num_sparse_fields = params['num_sparse_fields']

    learning_rate = params.get('learning_rate', 0.001)

    num_factors = params.get('num_factors', 4)
    deep_layers = params.get('deep_layers', [100, 100])

    model_type = params.get('model_type', 'PNN')

    dense_indices = tf.reshape(features['dense_indices'], shape=[-1, num_dense_fields])
    dense_values = tf.reshape(features['dense_values'], shape=[-1, num_dense_fields])

    sparse_indices = tf.reshape(features['sparse_indices'], shape=[-1, num_sparse_fields])
    sparse_values = tf.reshape(features['sparse_values'], shape=[-1, num_sparse_fields])

    num_features = num_dense_features + num_sparse_features
    num_fields = num_dense_fields + num_sparse_fields

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, shape=[-1, 1])

    with tf.variable_scope('lz_part'):
        # num_features * num_factors
        w = tf.get_variable('w', shape=[num_features, num_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        # batch * num_fields * num_factors
        embeddings = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * num_factors
        lz_part_matrix = tf.multiply(embeddings, tf.reshape(values, shape=[-1, num_fields, 1]))
        # batch * (num_fields * num_factors)
        lz_part = tf.reshape(lz_part_matrix, shape=[-1, num_fields * num_factors])

    with tf.variable_scope('IPNN_part'):
        # batch * num_fields * num_fields
        inner_product = tf.matmul(lz_part_matrix, tf.transpose(lz_part_matrix, perm=[0, 2, 1]))
        # batch * (num_fields * num_fields)
        inner_product = tf.reshape(inner_product, shape=[-1, num_fields * num_fields])

    with tf.variable_scope('OPNN_part'):
        outer_collection = []
        for i in range(num_fields):
            for j in range(i+1, num_fields):
                # batch * num_factors
                vi = tf.gather(lz_part_matrix, indices=i, axis=1)
                # batch * num_factors
                vj = tf.gather(lz_part_matrix, indices=j, axis=1)
                # batch * (num_factors * num_factors)
                outer_collection.append(tf.reshape(tf.einsum('ai,aj->aij', vi, vj), shape=[-1, num_factors * num_factors]))
        outer_product = tf.concat(outer_collection, axis=1)

    with tf.variable_scope('full_connection_part'):
        if model_type == 'IPNN':
            fc = tf.concat([lz_part, inner_product], axis=1)
        if model_type == 'OPNN':
            fc = tf.concat([lz_part, outer_product], axis=1)
        if model_type == 'PNN':
            fc = tf.concat([lz_part, inner_product, outer_product], axis=1)
        for i in range(len(deep_layers)):
            if i == 0:
                dense = tf.layers.dense(fc, deep_layers[i], activation=tf.nn.relu)
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