#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@file: fibinet_predict.py
@function:
@modify:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from fibinet import fibinet_model
from utils.parse_input import parse_input_v2


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)


def predict(num_factors, deep_layers, reduce_ratio):

    ret = parse_input_v2('../datasets/train.csv', '../datasets/test.csv', use_cross=True)
    ids = ret['test'][0]
    dense_indices = ret['test'][1]
    dense_values = ret['test'][2]
    sparse_indices = ret['test'][3]
    sparse_values = ret['test'][4]
    num_dense_fields = ret['num_fields'][0]
    num_sparse_fields = ret['num_fields'][1]
    num_dense_features = ret['num_features'][0]
    num_sparse_features = ret['num_features'][1]

    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(({'dense_indices': dense_indices,
                                                  'dense_values': dense_values,
                                                  'sparse_indices': sparse_indices,
                                                  'sparse_values': sparse_values}, ids))
        return ds

    def predict_input_fn():
        return input_fn()

    estimator = tf.estimator.Estimator(
        model_fn=fibinet_model.model_fn,
        model_dir='./model',
        params={
            'num_dense_features': num_dense_features,
            'num_sparse_features': num_sparse_features,
            'num_dense_fields': num_dense_fields,
            'num_sparse_fields': num_sparse_fields,
            'num_factors': num_factors,
            'deep_layers': deep_layers,
            'reduce_ratio': reduce_ratio
        }
    )

    preds = estimator.predict(predict_input_fn)

    return list(zip(ids, preds))


if __name__ == "__main__":
    res = predict(num_factors=16, deep_layers=[100, 100], reduce_ratio=4.0)
    for i in res:
        print(i)