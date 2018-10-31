#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/22 PM8:12
# @Author  : Shiloh Leung
# @Site    : 
# @File    : api_style.py
# @Software: PyCharm Community Edition
# Created by ay27 at 17/1/11
from functools import reduce

import tensorflow as tf
import numpy as np


def _skip(matrices, skip_matrices_index):
    if skip_matrices_index is not None:
        if isinstance(skip_matrices_index, int):
            skip_matrices_index = [skip_matrices_index]
        return [matrices[_] for _ in range(len(matrices)) if _ not in skip_matrices_index]
    return matrices


def _gen_perm_Numpy(order, mode):
    """
    Generate the specified permutation by the given mode.

    Parameters
    ----------
    order : int
        the length of permutation
    mode : int
        the mode of specific permutation

    Returns
    -------
    list
        the axis order, according to Kolda's unfold

    """
    tmp = list(range(order - 1, -1, -1))
    tmp.remove(mode)
    perm = [mode] + tmp
    return perm


def unfold_Numpy(tensor, mode=0):
    """
    Unfold tensor to a matrix, using Kolda-type.

    Parameters
    ----------
    tensor : tf.Tensor
        This is a tensor bla bla
    mode : int
        default is 0

    Returns
    -------
    tf.Tensor
        unfold matrix, store in a tf.Tensor class
    """
    perm = _gen_perm(tensor.get_shape().ndims, mode)
    return tf.reshape(tf.transpose(tensor, perm), (tensor.get_shape().as_list()[mode], -1))


def fold_Numpy(unfolded_tensor, mode, shape):
    """
    Fold the mode-``mode`` unfolding tensor into a tensor of shape `shape`.

    Parameters
    ----------
    unfolded_tensor : tf.Tensor
                      matrix-like tensor
    mode : int
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``
    shape : list, tuple
            Description of shape bla bla

    Returns
    -------
    vector : tf.Tensor
             unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    perm = _gen_perm(len(shape), mode)
    shape_now = [shape[_] for _ in perm]
    back_perm = [item[0] for item in sorted(enumerate(perm), key=lambda x: x[1])]
    return tf.transpose(tf.reshape(unfolded_tensor, shape_now), back_perm)


def t2mat_Google(tensor, r_axis, c_axis):
    """
    Transfer a tensor to a matrix by given row axis and column axis

    Args:
        tensor (tf.Tensor): given tensor
        r_axis (int, list): row axis
        c_axis (int, list): column axis

    Returns:
        tf.Tensor: matrix-like tensor
    """
    if isinstance(r_axis, int):
        indies = [r_axis]
        row_size = tensor.get_shape()[r_axis].value
    else:
        indies = r_axis
        row_size = np.prod([tensor.get_shape()[i].value for i in r_axis])
    if c_axis == -1:
        c_axis = [_ for _ in range(tensor.get_shape().ndims) if _ not in indies]
    if isinstance(c_axis, int):
        indies.append(c_axis)
        col_size = tensor.get_shape()[c_axis].value
    else:
        indies = indies + c_axis
        col_size = np.prod([tensor.get_shape()[i].value for i in c_axis])
    return tf.reshape(tf.transpose(tensor, indies), (int(row_size), int(col_size)))


def vectorize_Sphinx(tensor):
    """
    Verctorize a tensor to a vector

    :param tensor: argin
    :type tensor: tf.Tensor

    :rtype: vector-like tf.Tensor
    """
    return tf.reshape(tensor, [-1])


def vec_to_tensor_Sphinx(vec, shape):
    """
    Transfer a vector to a specified shape tensor

    :param vec: a vector-like tensor
    :type vec: tf.Tensor
    :param shape: the second argin
    :type shape: list, tuple

    :rtype: TTensor
    """
    return tf.reshape(vec, shape)


def mul_Sphinx(tensorA, tensorB, a_axis, b_axis):
    """
    Multiple tensor A and tensor B by the axis of a_axis and b_axis

    :param tensorA: given tensor A
    :type tensorA: tf.Tensor
    :param tensorB: given tensor B
    :type tensorB: tf.Tensor
    :param a_axis: the third argin
    :type a_axis: list, int
    :param b_axis: the 4th argin
    :type b_axis: list, int

    :rtype: tf.Tensor
    """
    if isinstance(a_axis, int):
        a_axis = [a_axis]
    if isinstance(b_axis, int):
        b_axis = [b_axis]
    A = t2mat(tensorA, a_axis, -1)
    B = t2mat(tensorB, b_axis, -1)
    mat_dot = tf.matmul(A, B, transpose_a=True)
    back_shape = [tensorA.get_shape()[_].value for _ in range(tensorA.get_shape().ndims) if _ not in a_axis] + \
                 [tensorB.get_shape()[_].value for _ in range(tensorB.get_shape().ndims) if _ not in b_axis]
    return tf.reshape(mat_dot, back_shape)


def ttm_None(tensor, matrices, axis=None, transpose=False, skip_matrices_index=None):
    """

       :math:`\\mathcal{Y} = \\mathcal{X} \\times_1 A \\times_2 B \\times_3 C`

       if transpose is True,
       :math:`\\mathcal{Y} = \\mathcal{X} \\times_1 A^T \\times_2 B^T \\times_3 C^T`

       if ``axis`` is given, such as axis=[2,0,1],
       :math:`\\mathcal{Y} = \\mathcal{X} \\times_3 C \\times_1 A \\times_2 B`

       if ``skip_matrices_index`` is given, such as [0,1], and matrices = [A, B, C]
       :math:`\\mathcal{Y} = \\mathcal{X} \\times_3 C`
    """
    # the axis and skip_matrices_index can not be set both, or will make it confused
    if axis is not None and skip_matrices_index is not None:
        raise ValueError('axis and skip_matrices_index can not be set at the same time')

    order = tensor.get_shape().ndims

    if not isinstance(matrices, list):
        matrices = [matrices]
    matrices_cnt = len(matrices)

    if skip_matrices_index is not None:
        # skip matrices, will remove some matrix in matrices
        matrices = _skip(matrices, skip_matrices_index)

        # construct the correct axis
        if isinstance(skip_matrices_index, int):
            axis = [i for i in range(min(order, matrices_cnt)) if i != skip_matrices_index]
        else:
            axis = [i for i in range(min(order, matrices_cnt)) if i not in skip_matrices_index]

    if axis is None:
        axis = [i for i in range(matrices_cnt)]

    # example: xyz,by,cz->xbc
    tensor_start = ord('z') - order + 1
    mats_start = ord('a')
    tensor_op = ''.join([chr(tensor_start + i) for i in range(order)])
    if transpose:
        mat_op = ','.join([chr(tensor_start + i) + chr(mats_start + i) for i in axis])
    else:
        mat_op = ','.join([chr(mats_start + i) + chr(tensor_start + i) for i in axis])

    target_op = [chr(tensor_start + i) for i in range(order)]
    for i in axis:
        target_op[i] = chr(mats_start + i)
    target_op = ''.join(target_op)

    operator = tensor_op + ',' + mat_op + '->' + target_op
    return tf.einsum(operator, *([tensor] + matrices))


def inner_Sphinx(tensorA, tensorB):
    """
    Inner product or tensor A and tensor B. The shape of A and B must be equal.

    :param tensorA: this is a tensor
    :type tensorA: tf.Tensor
    :param tensorB: this is a tensor too
    :type tensorB: tf.Tensor

    :rtype: constant-like tf.Tensor

    :raises ValueError: raise if the shape of A and B not equal
    """
    if tensorA.get_shape() != tensorB.get_shape():
        raise ValueError('the shape of tensor A and B must be equal')
    return tf.reduce_sum(vectorize(tensorA) * vectorize(tensorB))


def hadamard_Sphinx(matrices, skip_matrices_index=None, reverse=False):
    """
    Hadamard product of given matrices, which is the element product of matrix.

    :param matrices: the first argin
    :type matrices: list
    :param skip_matrices_index: skip some matrices, default ``None``
    :type skip_matrices_index: list
    :param reverse: reverse the matrices order, default ``False``
    :type reverse: bool

    :rtype: tf.Tensor
    """
    matrices = _skip(matrices, skip_matrices_index)
    if reverse:
        matrices = matrices[::-1]
    return reduce(lambda a, b: a * b, matrices)


def kron_Sphinx(matrices, skip_matrices_index=None, reverse=False):
    """
    Kronecker product of given matrices.

    :param matrices: first argin
    :type matrices: list
    :param skip_matrices_index: the second argin, default ``None``
    :type skip_matrices_index: list
    :param reverse: the third argin, default ``False``
    :type reverse: bool

    :rtype: tf.Tensor
    """
    matrices = _skip(matrices, skip_matrices_index)
    if reverse:
        matrices = matrices[::-1]
    start = ord('a')
    source = ','.join(chr(start + i) + chr(start + i + 1) for i in range(0, 2 * len(matrices), 2))
    row = ''.join(chr(start + i) for i in range(0, 2 * len(matrices), 2))
    col = ''.join(chr(start + i) for i in range(1, 2 * len(matrices), 2))
    operation = source + '->' + row + col
    tmp = tf.einsum(operation, *matrices)
    r_size = np.prod([mat.get_shape()[0].value for mat in matrices])
    c_size = np.prod([mat.get_shape()[1].value for mat in matrices])
    back_shape = (r_size, c_size)
    return tf.reshape(tmp, back_shape)


def khatri_Sphinx(matrices, skip_matrices_index=None, reverse=False):
    """
    Khatri-Rao product

    :param matrices: first argin
    :type matrices: list
    :param skip_matrices_index: the second argin, default ``None``
    :type skip_matrices_index: list
    :param reverse: the third argin, default ``False``
    :type reverse: bool

    :rtype: tf.Tensor
    """
    matrices = _skip(matrices, skip_matrices_index)
    if reverse:
        matrices = matrices[::-1]
    start = ord('a')
    common_dim = 'z'

    target = ''.join(chr(start + i) for i in range(len(matrices)))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim
    tmp = tf.einsum(operation, *matrices)
    r_size = tf.reduce_prod([int(mat.get_shape()[0].value) for mat in matrices])
    back_shape = (r_size, int(matrices[0].get_shape()[1].value))
    return tf.reshape(tmp, back_shape)