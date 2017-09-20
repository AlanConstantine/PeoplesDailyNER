# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date  : 2017-09-20 09:50:08

# import json
import numpy as np

# f = open(r'PDdatatest.json', 'r')
# data = json.load(f)

# labels = np.array(data['labels'])
# for i in labels:
#     print(np.array(i).shape)
# print(type(data['labels']))

# labels = np.array([(np.array(ele)) for ele in data['labels']]).ravel()
# print(labels.shape)
# for i in labels:
# assert (i.shape[-1] == 0)
# print(i.shape[0])
# print(labels[0])
labels = [[1, 2, 3], [1, 5, 3, 4]]
labels = np.array([(np.array(ele)).reshape(-1, 1) for ele in labels])


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


print(to_categorical(labels))
