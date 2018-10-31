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


def _gen_perm(order, mode):
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

    Examples
    --------
    >>> perm = _gen_perm(6, 2)
    list([2, 5, 4, 3, 1, 0])
    """
    tmp = list(range(order - 1, -1, -1))
    tmp.remove(mode)
    perm = [mode] + tmp
    return perm


def unfold(tensor, mode=0):
    """
    Unfold tensor to a matrix, using Kolda-type.

    Parameters
    ----------
    tensor : tf.Tensor
        the tensor with full shape
    mode : int
        default is 0, mode-``mode`` unfold

    Returns
    -------
    tf.Tensor
        a matrix-shape of unfolding tensor, store in a tf.Tensor class

    Examples
    --------
    >>> import tensorD.base.ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> tensor = tf.constant(np.arange(24).reshape(3,4,2))
    >>> unfolded_matrix = ops.unfold(tensor, 1)
    >>> tf.Session().run(unfolded_matrix)
    array([[ 0,  8, 16,  1,  9, 17],
       [ 2, 10, 18,  3, 11, 19],
       [ 4, 12, 20,  5, 13, 21],
       [ 6, 14, 22,  7, 15, 23]])

    """
    with tf.name_scope('unfold') as scope:
        perm = _gen_perm(tensor.get_shape().ndims, mode)
        return tf.reshape(tf.transpose(tensor, perm), (tensor.get_shape().as_list()[mode], -1))


def fold(unfolded_tensor, mode, shape):
    """Fold the mode-``mode`` unfolding tensor into a tensor of specific shape ``shape``.

    Parameters
    ----------
    unfolded_tensor : tf.Tensor
        matrix-shape tensor
    mode : int
        default is 0, indexing starts at 0, therefore mode is in ``range(0, len(shape))``
    shape : list, tuple
        the shape of folded tensor

    Returns
    -------
    tf.Tensor
        a full tensor of shape ``shape``

    Examples
    --------
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> import tensorD.base.ops as ops
    >>> tensor = tf.constant(np.arange(24).reshape(2,3,4))
    >>> unfolded_tensor = ops.unfold(tensor, 1)
    >>> folded_tensor = ops.fold(unfolded_tensor, 1, (2,3,4))
    >>> tf.Session().run(folded_tensor)
    array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
    """
    with tf.name_scope('fold') as scope:
        perm = _gen_perm(len(shape), mode)
        shape_now = [shape[_] for _ in perm]
        back_perm = [item[0] for item in sorted(enumerate(perm), key=lambda x: x[1])]
        return tf.transpose(tf.reshape(unfolded_tensor, shape_now), back_perm)


def t2mat(tensor, r_axis, c_axis):
    """
    Flat the tensor according to ``r_axis`` and ``c_axis``. Should be careful of the order
    of r_axis and c_axis.

    Parameters
    ----------
    tensor : tf.Tensor
        full-shape tensor
    r_axis : int, list
        row axis, must be equal or greater than 0
    c_axis : int, list
        column axis. If given as -1, the column axis is inferred from the remaining axis.

    Returns
    -------
    tf.Tensor
        matrix-shape of tensor

    Examples
    --------
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> import tensorD.base.ops as ops
    >>> tensor = tf.constant(np.arange(24).reshape(2,3,4))
    >>> mat1 = ops.t2mat(tensor, 1, -1)     # matrix shape is 3x(2*4)=3x8
    >>> mat2 = ops.t2mat(tensor, [0,2], -1) # matrix shape is (2*4)x3=8x3
    >>> mat3 = ops.t2mat(tensor, 1, [2, 0]) # Kolda-type mode-2 unfolding
    >>> mat4 = ops.t2mat(tensor, 1, [0, 2]) # LMV-type mode-2 unfolding

    """
    with tf.name_scope('t2mat') as scope:
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


def vectorize(tensor):
    """
    Reshape a tensor to a vector.

    Parameters
    ----------
    tensor : tf.Tensor
        full-shape of tensor

    Returns
    -------
    tf.Tensor
        vector-shape tensor

    Examples
    --------
    >>> vec = ops.vectorize(tf.constant(np.arange(24).reshape(3,4,2)))
    >>> tf.Session().run(vec)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23])
    """
    return tf.reshape(tensor, [-1], name='vectorize')


def vec_to_tensor(vec, shape):
    """
    Reshape a vector to a full tensor with specific shape.

    Parameters
    ----------
    vec : tf.Tensor
        vector-shape of tensor
    shape : list, tuple
        full tensor shape

    Returns
    -------
    tf.Tensor
        full tensor with shape ``shape``
    """
    return tf.reshape(vec, shape, name='vec2tensor')


def mul(tensorA, tensorB, a_axis, b_axis):
    """
    Multiple two tensors along given axis. The same definition with tensor contraction.
    See [1]_

    Parameters
    ----------
    tensorA : tf.Tensor
        tensor A
    tensorB : tf.Tensor
        tensor B
    a_axis : int, list, tuple
        tensor A axis to contract
    b_axis : int, list, tuple
        tensor B axis to contract, the len of ``a_axis`` and ``b_axis`` must be equal.

    Returns
    -------
    tf.Tensor
        the contracted tensor

    Examples
    --------
    tensor contraction

    >>> A = tf.constant(np.random.rand(2,3,4,5,6))
    >>> B = tf.constant(np.random.rand(1,2,3,5,7))
    >>> C = ops.mul(A, B, [0,1],[1,2])      # shape of C is 4x5x6x1x4x7
    >>> D = ops.mul(A, B, [0,1,3], [1,2,3]) # shape of C is 4x6x1x7

    classical matrix multiple

    >>> A = tf.constant(np.random.rand(4,5))
    >>> B = tf.constant(np.random.rand(5,4))
    >>> C = ops.mul(A, B, 1, 0)             # same as AB
    >>> D = ops.mul(A, B, 0, 1)             # same as BA

    References
    ----------
    .. [1] Cichocki, Andrzej. "Era of big data processing: A new approach via tensor networks and tensor decompositions."
     arXiv preprint arXiv:1403.2048 (2014).
    """
    with tf.name_scope('contraction') as scope:
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


def ttm(tensor, matrices, axis=None, transpose=False, skip_matrices_index=None):
    """
    Default is :math:`\\mathcal{Y} = \\mathcal{X} \\times_1 A \\times_2 B \\times_3 C`

    if transpose is True,
    :math:`\\mathcal{Y} = \\mathcal{X} \\times_1 A^T \\times_2 B^T \\times_3 C^T`

    if ``axis`` is given, such as axis=[2,0,1],
    :math:`\\mathcal{Y} = \\mathcal{X} \\times_3 C \\times_1 A \\times_2 B`

    if ``skip_matrices_index`` is given, such as [0,1], and matrices = [A, B, C],
    :math:`\\mathcal{Y} = \\mathcal{X} \\times_3 C`

    Should be noticed the ``axis`` and ``skip_matrices_index`` can not be set in same time!

    Parameters
    ----------
    tensor : tf.Tensor
        full-shape tensor
    matrices : list
        matrices to contract the tensor
    axis : list
        according axis of tensor to contract matrices.
        If ``axis`` is None, in default it will be ``range(0, len(matrices))``
    transpose : bool
        if True, transpose all matrices
    skip_matrices_index : int, list
        skip one or more matrices in ``matrices``

    Returns
    -------
    tf.Tensor
        contracted tensor

    Raises
    ------
    ValueError
        if axis and skip_matrices_index are given both

    Examples
    --------
    >>> tensorA = tf.constant(np.arange(24).reshape(2,4,3), dtype=tf.float32)
    >>> mat1 = tf.constant(np.arange(10).reshape(5, 2), dtype=tf.float32)
    >>> mat2 = tf.constant(np.arange(24).reshape(6, 4), dtype=tf.float32)
    >>> mat3 = tf.constant(np.arange(21).reshape(7, 3), dtype=tf.float32)
    >>> mats = [mat1, mat2, mat3]
    >>> contracted1 = ops.ttm(tensorA, mats) # shape is 5x6x7
    >>> contracted2 = ops.ttm(tensorA, [mat2, mat1, mat3], axis=[1,0,2]) # same as above
    >>> contracted3 = ops.ttm(tensorA, mats, skip_matrices_index=1)  # shape is 5x4x7
    >>> tensorB = tf.constant(np.arange(210).reshape(5,6,7), dtype=tf.float32)
    >>> contracted4 = ops.ttm(tensorB, mats, transpose=True) # shape is 2x4x3

    """
    with tf.name_scope('ttm') as scope:
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


def inner(tensorA, tensorB):
    """
    Inner product of two tensors with same shape.

    Parameters
    ----------
    tensorA : tf.Tensor
    tensorB : tf.Tensor

    Returns
    -------
    tf.Tensor

    Raises
    ------
    ValueError
        raise if the shapes are not equal
    """
    with tf.name_scope('inner') as scope:
        if tensorA.get_shape() != tensorB.get_shape():
            raise ValueError('the shape of tensor A and B must be equal')
        return tf.reduce_sum(vectorize(tensorA) * vectorize(tensorB))


def hadamard(matrices, skip_matrices_index=None):
    """
    Hadamard product of given matrices, which is the element product of matrix.

    Parameters
    ----------
    matrices : list of tf.Tensor
        matrices in same shape
    skip_matrices_index : int, list
        skip one or more matrices in ``matrices``

    Returns
    -------
    tf.Tensor
        result of hadamard product
    """
    with tf.name_scope('hadamard') as scope:
        matrices = _skip(matrices, skip_matrices_index)
        return reduce(lambda a, b: a * b, matrices)


def kron(matrices, skip_matrices_index=None, reverse=False):
    """
    Kronecker product of given matrices.

    Parameters
    ----------
    matrices : list of tf.Tensor
        a list of matrix-shape tensor
    skip_matrices_index : int, list
        skip one or more matrices in ``matrices``
    reverse : bool
        reverse matrices order

    Returns
    -------
    tf.Tensor
        a big matrix of kronecker product result
    """
    with tf.name_scope('kronecker') as scope:
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


def khatri(matrices, skip_matrices_index=None, reverse=False):
    """
    Khatri-Rao product of given matrices.

    Parameters
    ----------
    matrices : list of tf.Tensor
        a list of matrix-shape tensor, the column size must be equal
    skip_matrices_index : int, list
        skip one or more matrices
    reverse : bool
        reverse matrices order

    Returns
    -------
    tf.Tensor
        a matrix of khatri-rao result

    """
    with tf.name_scope('khatri-rao') as scope:
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

def max_single_value_mul(matrices, skip_matrices_index=None):
    """
        Product of max single values of given matrices.

        Parameters
        ----------
        matrices : list of tf.Tensor
            a list of matrix-shape tensor
        skip_matrices_index : int, list
            skip one or more matrices

        Returns
        -------
        tf.Tensor

    """
    with tf.name_scope('max-single-value-mul') as scope:
        matrices = _skip(matrices, skip_matrices_index)
        max_single_value = [tf.reduce_max(tf.svd(mat, compute_uv=False)) for mat in matrices]
        return reduce(lambda a, b: a * b, max_single_value)





def xcb(X, C, B):
    X = unfold(X, 0)
    I = X.get_shape()[0].value
    J = B.get_shape()[0].value
    K = C.get_shape()[0].value
    R = C.get_shape()[1].value

    I1 = tf.ones((I, 1), X.dtype)
    J1 = tf.ones((J, 1), X.dtype)
    K1 = tf.ones((K, 1), X.dtype)
    JK1 = tf.ones((J * K, 1), X.dtype)

    binX = tf.cast(tf.not_equal(X, tf.zeros(X.get_shape(), X.dtype)), X.dtype)

    res = []

    for r in range(R):
        Cr = tf.slice(C, [0, r], [K, 1])
        Br = tf.slice(B, [0, r], [J, 1])
        N1 = hadamard([X, tf.matmul(I1, kron([Cr, J1]), transpose_b=True)])
        N2 = hadamard([binX, tf.matmul(I1, kron([K1, Br]), transpose_b=True)])
        N3 = hadamard([N1, N2])

        res.append(tf.matmul(N3, JK1))
    return tf.reshape(tf.stack(res, axis=1), (I, R))
