# # Created by ay27 at 17/2/7
# import parse
#
import csv
import numpy as np
import tensorflow as tf
from ctypes import ArgumentError


class TensorReader(object):
    """

    Attibutes
    ---------

    """

    def __init__(self, file_path):
        """

        Parameters
        ----------
        file_path: file path
        type: type of file, default 'csv'
        """
        self._file_path = file_path
        self._dense = None
        self._sparse_data = None
        self._full_data = None
        self._type = self._file_path.split('.')[-1]

    def read(self, full_shape=None):
        file = open(self._file_path, 'r')
        str_in = []
        if self._type == 'csv' or self._type == 'txt':
            for row in csv.reader(file):
                if len(row) != 0:
                    str_in.append(row)
        else:
            raise ArgumentError(self._type + ' file is not supported by TensorReader.')
        file.close()

        order = len(str_in[0]) - 1
        entry_count = len(str_in)
        value = np.zeros(entry_count)
        idx = np.zeros(shape=(entry_count, order), dtype=int)
        for row in range(entry_count):
            entry = str_in[row]
            idx[row] = np.array([int(entry[mode]) for mode in range(order)])
            value[row] = float(entry[-1])

        if full_shape==None:
            max_dim = np.max(idx, axis=0) + np.ones(order).astype(int)
        else:
            max_dim = full_shape

        self._sparse_data = tf.SparseTensor(indices=idx, values=value, dense_shape=max_dim)
        self._full_data = tf.sparse_tensor_to_dense(self._sparse_data, validate_indices=False)

    @property
    def full_data(self):
        return self._full_data

    @property
    def sparse_data(self):
        return self._sparse_data
