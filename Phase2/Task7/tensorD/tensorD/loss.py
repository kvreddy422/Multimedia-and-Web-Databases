# Created by ay27 at 17/2/23
import tensorflow as tf
from tensorD.base import *


def l2(f, h):
    """
    L2 norm

    loss = \\frac{1}{2} ||f - h||^2_2
    Parameters
    ----------
    f : tf.Tensor
    h : tf.Tensor

    Returns
    -------
    tf.Tensor
        a tensor hold the scalar value

    """
    with tf.name_scope('L2-norm') as scope:
        return 0.5 * tf.reduce_sum(tf.square(f - h))


def rmse(A, B):
    """
    Root Mean Square Error.

    RMSE = \\sqrt{ \\frac{1}{c} \\left \\| A - B  \\right \\|_F^2}
         = \\sqrt{ \\frac{1}{c} \\sum (A_{ij} - B_{ij})^2}

    Parameters
    ----------
    A : tf.Tensor
        origin tensor
    B : tf.Tensor
        hypothesis tensor with error

    Returns
    -------
    tf.Tensor
        RMSE scalar value

    Raises
    ------
    TensorErr
        if the shape of A and B are not equal

    """
    with tf.name_scope('RMSE') as scope:
        if A.get_shape() != B.get_shape():
            raise TensorErr('the shape of tensor A and B must be equal')
        diff_tensor = tf.subtract(A, B)
        return tf.sqrt(tf.reduce_sum(tf.square(diff_tensor)) / diff_tensor.get_shape().num_elements())


def rmse_ignore_zero(A, B):
    """
    Root Mean Square Error, ignore zero element in tensor A.

    RMSE = \sqrt{ \frac{1}{c} \left \| A - B  \right \|_F^2}
         = \sqrt{ \frac{1}{c} \sum (A_{ij} - B_{ij})^2}

    Parameters
    ----------
    A : tf.Tensor
        origin tensor
    B : tf.Tensor
        hypothesis tensor with error

    Returns
    -------
    tf.Tensor
        RMSE scalar value

    Raises
    ------
    TensorErr
        if the shape of A and B are not equal

    """
    with tf.name_scope('RMSE-ignore-zero') as scope:
        if A.get_shape() != B.get_shape():
            raise TensorErr('the shape of tensor A and B must be equal')
        B = B * tf.cast(tf.not_equal(A, 0), B.dtype)
        diff_tensor = tf.subtract(A, B)
        return tf.sqrt(tf.reduce_sum(tf.square(diff_tensor)) / tf.reduce_sum(tf.cast(tf.not_equal(A, 0), B.dtype)))
