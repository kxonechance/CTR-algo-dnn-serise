#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@file: fnn_model.py
@function:
@modify:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build Factorisation Machine supported Neural Network(FNN) model with binary_classification_head."""

    num_dense_features = params['num_dense_features']
    num_sparse_features = params['num_sparse_features']
    num_dense_fields = params['num_dense_fields']
    num_sparse_fields = params['num_sparse_fields']

    learning_rate = params.get('learning_rate', 0.001)

    num_factors = params.get('num_factors', 4)
    deep_layers = params.get('deep_layers', [100, 100])

    dense_indices = tf.reshape(features['dense_indices'], shape=[-1, num_dense_fields])
    dense_values = tf.reshape(features['dense_values'], shape=[-1, num_dense_fields])

    sparse_indices = tf.reshape(features['sparse_indices'], shape=[-1, num_sparse_fields])
    sparse_values = tf.reshape(features['sparse_values'], shape=[-1, num_sparse_fields])

    num_features = num_dense_features + num_sparse_features
    num_fields = num_dense_fields + num_sparse_fields

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, shape=[-1, 1])

    with tf.variable_scope('linear_part'):
        # num_features * 1
        w = tf.get_variable('w', shape=[num_features, 1], initializer=tf.initializers.glorot_normal())
        # bias
        # 1 * 1
        b = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.0))
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        # batch * num_fields * 1
        embeddings = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * 1
        linear_part = tf.multiply(embeddings, tf.reshape(values, shape=[-1, num_fields, 1]))
        # batch * 1
        linear_part = tf.reduce_sum(linear_part, axis=1)
        # batch * 1
        b = b * tf.ones_like(linear_part, dtype=tf.float32)
        # batch * 1
        linear_part = linear_part + b

    with tf.variable_scope('deep_part'):
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
        # batch * (num_fields * num_factors)
        embeddings = tf.reshape(embeddings, shape=[-1, num_fields*num_factors])
        # deep_part
        for i in range(len(deep_layers)):
            if i == 0:
                dense = tf.layers.dense(embeddings, deep_layers[i], activation=tf.nn.relu)
                dense = tf.layers.batch_normalization(dense, training=mode == tf.estimator.ModeKeys.TRAIN)
                dense = tf.layers.dropout(dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
            else:
                dense = tf.layers.dense(dense, deep_layers[i], activation=tf.nn.relu)
                dense = tf.layers.batch_normalization(dense, training=mode == tf.estimator.ModeKeys.TRAIN)
                dense = tf.layers.dropout(dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('output'):
        # batch * 1
        logits = tf.layers.dense(dense, 1)
        # batch * 1
        logits = linear_part + logits

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