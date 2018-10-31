# Created by ay27 at 17/2/7
import numpy as np


class Provider(object):
    """
    Base Data Provider, should be noted that the batch size should be given in initializer.
    """

    @property
    def batch_size(self):
        raise NotImplementedError

    def full_tensor(self):
        """

        Returns
        -------
        tf.Tensor
            dense or sparse tensor hold the full data

        """
        raise NotImplementedError

    def data_queue(self, shuffled=True):
        """
        Parameters
        ----------
        shuffled : bool
            shuffle the queue data or not, default is **True**

        Returns
        -------
        tf.Tensor
            a data queue to read data continuous according to the **fix** batch size

        """
        raise NotImplementedError


# class OrdProvider(Provider):
#     """
#     Data Provider, split data in given order(mode).
#     """
#
#     def __init__(self, reader, order, task_cnt=1, task_index=0, batch_size=1, sparse=False, shape=None):
#         self.reader = reader
#         self.order = order
#         self.task_index = task_index
#         self.task_cnt = task_cnt
#         self.is_sparse = sparse
#         self.shape = shape
#         self.batch_size = batch_size
#
#         # store in dense
#         self.data = None
#
#         self._split_size = None
#
#         self._offset = None
#
#         if self.is_sparse:
#             self._read_sparse()
#         else:
#             self._read_dense()
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         """
#
#         Yields
#         ------
#         ndarray
#             batch of data
#         """
#         cur_index = 0
#         while cur_index < self._split_size:
#             end = min(cur_index + self.batch_size, self._split_size)
#             yield self.data[cur_index:end]
#             cur_index += self.batch_size
#         raise StopIteration()
#
#     def _read_sparse(self):
#         input_data = np.array([row for row in self.reader.next()])
#         if not self.shape:
#             self.shape = np.max(input_data, axis=0)[:self.order]
#         for _ in range(self.order):
#             self.shape[_] = int(self.shape[_])
#
#         self._split_size = int(self.shape[self.order] / self.task_cnt)
#         self._offset = self.task_index * self._split_size
#
#         split_shape = self.shape.copy()
#         split_shape[self.order] = self._split_size
#         self.data = np.zeros(split_shape)
#         for row in input_data:
#             if self._offset <= row[self.order] < self._offset + self._split_size:
#                 row[self.order] -= self._offset
#                 self.data.itemset(row[:-1], row[-1])
#
#     def _read_dense(self):
#         self.data = np.asarray(
#             [row for (i, row) in enumerate(self.reader.next()) if
#              self._offset <= i < self._offset + self._split_size])
#         if not self.shape:
#             self.shape = self.data.shape
#
#         self._split_size = int(self.shape[self.order] / self.task_cnt)
#         self._offset = self.task_index * self._split_size
