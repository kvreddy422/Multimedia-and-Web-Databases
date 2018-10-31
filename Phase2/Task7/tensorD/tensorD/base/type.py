# Created by ay27 at 17/1/11
import numpy as np
import tensorflow as tf
from . import ops


class DTensor(object):
    """
    Dense Tensor
    """

    def __init__(self, tensor):
        """

        Parameters
        ----------
        tensor : tf.Tensor, ndarray

        """
        if isinstance(tensor, tf.Tensor):
            self.T = tensor
            self.shape = tensor.get_shape().as_list()
        else:
            self.T = tf.constant(tensor)
            self.shape = tensor.shape
        self.unfold_T = None
        self.fold_T = self.T

    def mul(self, tensor, a_axis=0, b_axis=0):
        """
        Multiple(contract) with DTensor along given axis.

        Parameters
        ----------
        tensor : DTensor
            must be a DTensor
        a_axis : int, list
            axis belong to self DTensor
        b_axis : int, list
            axis belong to given tensor

        Returns
        -------
        DTensor
            the result DTensor of contraction

        """
        return DTensor(ops.mul(self.T, tensor.T, a_axis, b_axis))

    def unfold(self, mode=0):
        if self.unfold_T is None:
            self.unfold_T = ops.unfold(self.T, mode)
        return DTensor(self.unfold_T)

    def t2mat(self, r_axis, c_axis):
        return DTensor(ops.t2mat(self.T, r_axis, c_axis))

    def vectorize(self):
        return DTensor(ops.vectorize(self.T))

    def get_shape(self):
        return self.T.get_shape()

    def kron(self, tensor):
        if isinstance(tensor, DTensor):
            return DTensor(ops.kron([self.T, tensor.T]))
        else:
            return DTensor(ops.kron([self.T, tensor]))

    def khatri(self, tensor):
        if isinstance(tensor, DTensor):
            return DTensor(ops.khatri([self.T, tensor.T]))
        else:
            return DTensor(ops.khatri([self.T, tensor]))

    def eval(self, feed_dict=None, session=None):
        return self.T.eval(feed_dict, session)

    @staticmethod
    def fold(unfolded_tensor, mode, shape):
        return DTensor(ops.fold(unfolded_tensor, mode, shape))

    def __add__(self, other):
        if isinstance(other, DTensor):
            return DTensor(self.T + other.T)
        else:
            return DTensor(self.T + other)

    def __mul__(self, other):
        """
        Hadamard product with other tensor, an element-wise product.

        Parameters
        ----------
        other : tf.Tensor, DTensor

        Returns
        -------
        DTensor
            result of hadamard product
        """
        if isinstance(other, DTensor):
            return DTensor(self.T * other.T)
        else:
            return DTensor(self.T * other)

    def __sub__(self, other):
        if isinstance(other, DTensor):
            return DTensor(self.T - other.T)
        else:
            return DTensor(self.T - other)

    def __getitem__(self, index):
        return self.T[index]


class KTensor(object):
    """
    Kruskal Tensor

    .. math::
        \\mathcal{X} = \\sum_r \\sigma_r a_r \\circ b_r \\circ c_r

    """

    def __init__(self, factors, lambdas=None):
        """
        Parameters
        ----------
        factors : list of [tf.Tensor, ndarray]
            the factor matrix of Kruskal Tensor
        lambdas : ndarray, tf.Tensor
            vector-like, the weight of every axis of factors
        """
        if isinstance(factors[0], np.ndarray):
            self.U = [tf.constant(mat, dtype=tf.float32) for mat in factors]
        else:
            self.U = factors

        # Note that the shape of lambdas must be (x, 1).
        # The dimension of "1" should not be ignored!!
        if lambdas is None:
            self.lambdas = tf.ones((self.U[0].get_shape()[1].value, 1), dtype=tf.float32)
        else:
            if isinstance(lambdas, np.ndarray):
                self.lambdas = tf.constant(lambdas)
            else:
                self.lambdas = lambdas
            if self.lambdas.get_shape().ndims == 1:
                self.lambdas = tf.expand_dims(self.lambdas, -1)


        self.order = len(self.U)

    def extract(self):
        """
        .. math::
            \\mathcal{X} = \\sum_r \\sigma_r a_r \\circ b_r \\circ c_r = (A \\odot B \\odot C) \\times \\Sigma

        Returns
        -------
        tf.Tensor
            full-shape tensor
        """
        with tf.name_scope('KTensor-extract') as scope:
            tmp = ops.khatri(self.U)
            back_shape = [U.get_shape()[0].value for U in self.U]
            return tf.reshape(tf.matmul(tmp, self.lambdas), back_shape)


class TTensor(object):
    """
    Tucker Tensor

    .. math::
        \\mathcal{X} =  \\mathcal{G} \\times_1 A \\times_2 B \\times_3 C

    """

    def __init__(self, core, factors):
        """
        Parameters
        ----------
        core : tf.Tensor, ndarray
            :math:`\\mathcal{G}`
        factors : List of tf.Tensor or ndarray
            factor matrix of Tucker Tensor
        """
        if isinstance(core, np.ndarray):
            self.g = tf.constant(core)
        else:
            self.g = core
        if isinstance(factors[0], np.ndarray):
            self.U = [tf.constant(mat) for mat in factors]
        else:
            self.U = factors
        self.order = self.g.get_shape().ndims

    def extract(self):
        """
        Extract the full tensor of core and factors.

        Returns
        -------
        tf.Tensor
            full-shape of Tensor
        """
        return ops.ttm(self.g, self.U)
