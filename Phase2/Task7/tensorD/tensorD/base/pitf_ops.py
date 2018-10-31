from functools import reduce
import tensorflow as tf
import numpy as np
import time
import unittest
from functools import reduce
import logging
import tensorD.base.ops as ops
from numpy.random import rand
# all tensor dtype is tf.float32


def generate(shape, rank):
    """
    Generate matrix randomly(use standard normal distribution) by given shape and rank.

    Parameters
    ----------
    shape: int
        2-dim tuple.
        First element in tuple is the number of rows of the matrix U.
        Second element in tuple is the number of rows of the matrix V.

    rank: int
        The rank of matrix.
        And it`s also the number of columns of matrix U and V.

    Returns
    -------
    u,v: the generated matrix U and V.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3,4])
    >>> rank = tf.constant(5)
    >>> U, V = pitf_ops.generate((3,4), 5)
    >>> tf.Session().run(U)
    >>> tf.Session().run(V)

    """
    u = tf.random_normal((shape[0], rank), name='random_normal_u')
    v = tf.random_normal((shape[1], rank), name='random_normal_v')
    return u, v


def centralization(mat):
    """
    This function makes matrix to be centralized.

    Parameters
    ----------
    mat:
    The uncentralized matrix

    Returns
    -------
    ctr_mat:
    The centralized matrix.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> mat = tf.random_normal((3,4))
    >>> ctrmat = pitf_ops.centralization(mat)
    >>> tf.Session().run(ctrmat)
    """
    shape = tf.shape(mat)
    tmp = tf.matmul(tf.ones((shape[0], shape[0]), dtype=tf.float32, name='ctr_ones'), mat, name='ctr_mul')/tf.cast(shape[0], dtype=mat.dtype)
    ctr_mat = tf.subtract(mat, tmp, name='ctf_sub')
    return ctr_mat


def subspace(shape, rank, mode=None):
    """
    Make the matrix A,B,C from pairwise interaction tensor satisfy the constraints and uniqueness.

    Parameters
    ----------
    shape: int 2-dim tuple.
    First element in tuple is the number of rows of the matrix U.
    Second element in tuple is the number of rows of the matrix V.

    rank: int
    The rank of matrix.
    And it`s also the number of columns of matrix U and V.

    mode: str
    Point out to use function for which matrix.
    mode option is 'A', 'B','C'.(default option is None.)

    Returns
    -------
    Psb,Psc:
    The matrix which satisfied the constraints.

    Psa:
    The matrix which satisfied the constraints.But it`s more complicated than Psb and Psc
    because of the constraints difference.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3,4])
    >>> rank = tf.constant(3)
    >>> result = pitf_ops.subspace(shape,rank,'A')
    >>> tf.Session().run(result)
    """
    U, V = generate(shape, rank)
    tmp = tf.matmul(U, V, transpose_a=False, transpose_b=True,name='subspace_mul')
    if mode == 'B':
        Psb = centralization(tmp)
        return Psb
    if mode == 'C':
        Psc = centralization(tmp)
        return Psc
    if mode == 'A':
        row = shape[0]
        col = shape[1]
        vec1 = tf.ones((row, 1))
        vec1_t = tf.transpose(vec1)
        vec2 = tf.ones((col, 1))
        vec2_t = tf.transpose(vec2)
        Psa = centralization(tmp)+tf.matmul(tf.matmul(vec1_t, tmp), vec2)*(vec1*vec2_t)/tf.cast((row*col), dtype=tmp.dtype)
        return Psa
    return False


def sample_rule4mat(shape, ra, rb, rc):
    """
    Generate the appointed matrix  A,B,C to pairwise interaction tensor by appointed shape
    and rank(ra, rb, rc).

    Parameters
    ----------
    shape: int 3-dim tuple.
    this is the tensor`s shape.

    rank(ra, rb, rc): int
    The rank of three different matrix A,B,C.

    Returns
    -------
    A,B,C:
    Three generated matrix by appointed shape and rank.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([6,7])
    >>> ra = tf.constant(3)
    >>> rb = tf.constant(4)
    >>> rc = tf,constant(5)
    >>> result = pitf_ops.sample_rule4mat(shape, ra, rb, rc)
    >>> tf.Session().run(result)
    """
    A = subspace((shape[0], shape[1]), ra, 'A')
    B = subspace((shape[1], shape[2]), rb, 'B')
    C = subspace((shape[2], shape[0]), rc, 'C')
    return A, B, C


def sample3D_rule(shape, sample_num):
    """
    Generate the rule of sampling from matrix  A,B,C.(following uniform distribution.)
    First this function produces three random list, and then combine pairwise to become
    the indices.

    Parameters
    ----------
    shape: int 3-dim tuple.
    This is the tensor`s shape.

    sample_num: int
    It`s the sampling number from tensor witch also become length of subscript list.

    Returns
    -------
    a,b,c:
    Three difference indeices will be used to sampling from matrices and tensor.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3,4,5])
    >>> sample_num = 10
    >>> a, b, c pitf_ops.sample3D_rule(shape, sample_num)
    >>> tf.Session().run(a)
    >>> tf.Session().run(b)
    >>> tf.Session().run(c)
    """
    a = tf.random_uniform([sample_num], 0, shape[0], dtype=tf.int32, name='sample_rule_a')
    b = tf.random_uniform([sample_num], 0, shape[1], dtype=tf.int32, name='sample_rule_b')
    c = tf.random_uniform([sample_num], 0, shape[2], dtype=tf.int32, name='sample_rule_c')
    # a = np.random.randint(0, shape[0], sample_num)
    # b = np.random.randint(0, shape[1], sample_num)
    # c = np.random.randint(0, shape[2], sample_num)
    return a, b, c


def Pomega_mat(spl, mat, shape, sample_num, dim=None):
    """
    Design operators from sampling list.

    Parameters
    ----------
    spl:
    Sample list includes three rows a, b, c.
    Pairwise combine a, b, c, and generate three sampling indices.

    mat:
    The sampled matrix.

    shape:
    The shape of tensor.

    sample_num: int
    It`s the sampling number from tensor witch also become length of subscript list.

    dim:
    The parameter dim controls dimension, then sample from different matrix by given
    subscript.

    Returns
    -------
    Pomega_AX,Pomega_BY,Pomega_CZ:
    Result is a sample vector whose length is sampling number by follow sampling rule.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3, 4, 5])
    >>> sp_num = 10
    >>> a, b, c = pitf_ops.sample3D_rule(shape, sp_num)
    >>> spl = [a, b, c]
    >>> mat1 = tf.random_normal((3, 4))
    >>> mat2 = tf.random_normal((4, 5))
    >>> mat3 = tf.random_normal((5, 3))
    >>> A = pitf_ops.Pomega_AX(spl, mat1, shape, sp_num, dim=0)
    >>> B = pitf_ops.Pomega_BY(spl, mat2, shape, sp_num, dim=1)
    >>> C = pitf_ops.Pomega_CZ(spl, mat3, shape, sp_num, dim=2)
    >>> tf.Session().run(A)
    >>> tf.Session().run(B)
    >>> tf.Session().run(C)
    """
    tmp = []
    if dim == 0:
        for i in range(sample_num):
            t1, t2 = spl[0][i], spl[1][i]
            tmp.append(mat[t1][t2])
        sum_vec = tmp
        Pomega_AX = tf.div(sum_vec, tf.sqrt(tf.cast(shape[2], dtype=mat.dtype)), name='Pomega_AX')
        # print('Pomega_AX')
        return Pomega_AX

    elif dim == 1:
        for i in range(sample_num):
            t2, t3 = spl[1][i], spl[2][i]
            tmp.append(mat[t2][t3])
        sum_vec = tmp
        Pomega_BY = tf.div(sum_vec, tf.sqrt(tf.cast(shape[0], dtype=mat.dtype)), name='Pomega_BY')
        # print('Pomega_BY')
        return Pomega_BY

    elif dim == 2:
        for i in range(sample_num):
            t3, t1 = spl[2][i], spl[0][i]
            tmp.append(mat[t3][t1])
        sum_vec = tmp
        Pomega_CZ = tf.div(sum_vec, tf.sqrt(tf.cast(shape[1], dtype=mat.dtype)), name='Pomega_CZ')
        # print('Pomega_CZ')
        return Pomega_CZ

    else:
        raise ValueError


def index_value_append(spl, sp_num, sample_vec, l1=0, l2=0):
    """
    This function generates two lists: Indices and Values.
    Indices is used to record sampling position.
    Values is used to record sampling value from sample vector.

    Parameters
    ----------
    spl:
    Sample list includes three rows a, b, c.
    Pairwise combine a, b, c, and generate three sampling indices.

    sp_num:int
    The sample number.

    sample_vec:
    The shape of tensor.

    l1,l2:
    Control the subscript from sampling list to choose.

    Returns
    -------
    indices:
    Indices is used to record sampling position.
    values:
    Values is used to record sampling value from sample vector.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3, 4, 5])
    >>> sp_num = 10
    >>> a, b, c = pitf_ops.sample3D_rule(shape, sp_num)
    >>> spl = [a, b, c]
    >>> sample_vec = tf.constant([1,2,3,4,5,6,7,8,9,0])
    >>> indices, values = pitf_ops.index_value_append(spl, sp_num, sample_vec, 0, 0)
    """
    indices = []
    values = []
    for i in range(sp_num):
        indices.append([spl[l1][i], spl[l2][i]])
        values.append(sample_vec[i])
    return indices, values


def adjoint_operator(spl, sample_vec, shape, sp_num, dim=None):
    """
    The adjoint operator of operator (function Pomega_mat).

    Parameters
    ----------
    spl:
    Sample list includes three rows a, b, c.
    Pairwise combine a, b, c, and generate three sampling indices.

    sample_vec:
    The shape of tensor.

    shape:
    The shape of tensor.

    sp_num:int
    The sample number.

    dim:
    The parameter dim controls dimension, then sample from different matrix by given
    subscript.

    Returns
    -------
    mat_zero+dense_mat :
    Return a matrix.Note that the matrix is sparse, and intermediate specific comutation in
    this function is little complicated ,which force us to convert tf.Tensor to
    tf.SparseTensor, and later restore dense form back.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3, 4, 5])
    >>> sp_num = 10
    >>> sp_vec = tf.random_uniform([sp_num])
    >>> X = pitf_ops.adjoint_operator(spl, sp_vec, shape, sp_num, dim=0)
    >>> Y = pitf_ops.adjoint_operator(spl, sp_vec, shape, sp_num, dim=1)
    >>> Z = pitf_ops.adjoint_operator(spl, sp_vec, shape, sp_num, dim=2)
    >>> tf.Session().run(X)
    >>> tf.Session().run(Y)
    >>> tf.Session().run(Z)
    """
    if dim == 0:
        mat_zero = tf.zeros((shape[0], shape[1]), dtype=sample_vec.dtype)
        # print('dim0 ready loop')
        indices, values = index_value_append(spl, sp_num, sample_vec, 0, 1)
        # print('loop done.')
        indices = tf.cast(indices, dtype=tf.int64)
        values = tf.cast(values, dtype=sample_vec.dtype)
        shape_t = tf.cast(shape, dtype=tf.int64)
        mat_shape = shape_t[0:2]
        # tensor = tf.sparse_to_dense(indices, mat_shape, value)
        # print('ready to convert')
        delta = tf.SparseTensor(indices, values, mat_shape)
        dense_mat = tf.sparse_tensor_to_dense(delta, validate_indices=False, name='adjoint_s2d_1')
        # print('convert done.')
        # print('adjoint_operator 1')
        return mat_zero+dense_mat
    elif dim == 1:
        mat_zero = tf.zeros((shape[1], shape[2]), dtype=sample_vec.dtype)
        indices, values = index_value_append(spl, sp_num, sample_vec, 1, 2)
        indices = tf.cast(indices, dtype=tf.int64)
        values = tf.cast(values, dtype=sample_vec.dtype)
        shape_t = tf.cast(shape, dtype=tf.int64)
        mat_shape = shape_t[1:3]
        # tensor = tf.sparse_to_dense(indices, mat_shape, value)
        delta = tf.SparseTensor(indices, values, mat_shape)
        dense_mat = tf.sparse_tensor_to_dense(delta, validate_indices=False,  name='adjoint_s2d_2')
        print('adjoint_operator 2')
        return mat_zero+dense_mat
    elif dim == 2:
        mat_zero = tf.zeros((shape[2], shape[0]), dtype=sample_vec.dtype)
        indices, values = index_value_append(spl, sp_num, sample_vec, 2, 0)
        indices = tf.cast(indices, dtype=tf.int64)
        values = tf.cast(values, dtype=sample_vec.dtype)
        shape_t = tf.cast(shape, dtype=tf.int64)
        shape_t_re = shape_t[::-1]
        mat_shape = shape_t_re[0:3:2]
        # tensor = tf.sparse_to_dense(indices, mat_shape, value)
        delta = tf.SparseTensor(indices, values, mat_shape)
        dense_mat = tf.sparse_tensor_to_dense(delta, validate_indices=False,  name='adjoint_s2d_3')
        print('adjoint_operator 3')
        return mat_zero+dense_mat
    else:
        raise ValueError


def Pomega_tensor(spl, tensor, sp_num):
    """
    According to the result of sampling rule, this function samples from original tensor.

    Parameters
    ----------
    spl:
    Sample list includes three rows a, b, c.
    Pairwise combine a, b, c, and generate three sampling indices.

    tensor:
    The original tensor.

    sp_num:int
    The sample number.

    Returns
    -------
    A vector which includes tensor`s element based on sampling list following sampling rule.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3, 4, 5])
    >>> sp_num = 20
    >>> a, b, c = pitf_ops.sample3D_rule(shape, sp_num)
    >>> spl = [a, b, c]
    >>> tensor = tf.random_normal((3, 4, 5))
    >>> sp_t = pitf_ops.Pomega_tensor(spl, tensor, sp_num)
    >>> tf.Session().run(sp_t)
    """
    sp_t = []
    tensor_t = tensor
    print('Pomega_tensor loop')
    for i in range(sp_num):
        sp_t.append(tensor_t[spl[0][i]][spl[1][i]][spl[2][i]])
    print('Pomega_tensor')
    return tf.convert_to_tensor(sp_t, name='Pomega_tensor')


def Pomega_Pair(spl, X, Y, Z, tensor_shape, sp_num):
    """
    According to the result of sampling rule, this function samples from three matrices A,B,C.

    Parameters
    ----------
    spl:
    Sample list includes three rows a, b, c.
    Pairwise combine a, b, c, and generate three sampling indices.

    X,Y,Z:
    The three matrices.

    tensor_shape:
    The original tensor`s shape(3-dim tuple).

    sp_num:int
    The sample number.

    Returns
    -------
    Pomega_Pair:
    It generates a vector which is the sum of three different results computed by sampling
    function Pomega_mat.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>>	shape = tf.constant([3, 4, 5])
    >>> sp_num = 10
    >>>	a, b, c = sample3D_rule(shape, sp_num)
    >>>	spl = [a, b, c]
    >>>	mat1 = tf.random_normal((3, 4))
    >>>	mat2 = tf.random_normal((4, 5))
    >>>	mat3 = tf.random_normal((5, 3))
    >>>	shape = tf.constant([3, 4, 5])
    >>>	sp_num = 10
    >>> result = Pomega_Pair(spl, X, Y, Z, tensor_shape, sp_num
    >>> tf.Session().run(result)
    """
    Pomega_A = Pomega_mat(spl, X, tensor_shape, sp_num, 0)
    Pomega_B = Pomega_mat(spl, Y, tensor_shape, sp_num, 1)
    Pomega_C = Pomega_mat(spl, Z, tensor_shape, sp_num, 2)
    Pomega_pair = Pomega_A + Pomega_B + Pomega_C
    # print('Pomega_Pair')
    return Pomega_pair


def cone_projection_operator(x, t):
    """
    When sampling process with noise, we should design a cone projection operator to solve
    these noisy observations.
    The detail can consult the paper Exact and Stable Recovery of Pairwise Interaction Tensors.

    Parameters
    ----------
    x:
    A vector.

    t:
    A constant.

    Returns
    -------
    Resturn the new x, t as the result of computing cone projection operator.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>>	xx=tf.random_normal([5])
    >>>	tt=tf.constant(1)
    >>>	t1,t2=cone_projection_operator(xx,tt)
    >>> tf.Session().run(t1)
    """
    norm_x = tf.norm(x, ord='euclidean')
    if norm_x <= t:
        return x, t
    if t <= tf.negative(norm_x):
        return 0, 0
    if t >= tf.negative(norm_x) and t <= norm_x:
        tmp = (norm_x + t)/(2 * norm_x)
        return tmp*x, tmp*norm_x


def SVT(mat, tao):
    """
    Do singualr value thresholding.

    Parameters
    ----------
    mat:
    The matrix.

    tao:
    The threshold value.

    Returns
    -------
    return form is like to S,U,V(S is diagonal matrix,U is left singualr value matrix).

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3, 4, 5])
    >>> mat1 = tf.random_normal((3, 4))
    >>> tao = tf.constant(0.0)
    >>> s, u, v = SVT(mat1, tao)
    >>> tf.Session().run(s)
    """
    s, u, v = tf.svd(centralization(mat), full_matrices=True, compute_uv=True, name='svd')
    # length = tf.size(s)
    length = s.get_shape().as_list()
    s_t = []
    # print('svt loop')
    for i in range(length[0]):
        # the type of zero has to be same as the element type of s)
        s_t.append(tf.maximum(tf.cast(0, dtype=mat.dtype), s[i]-tao))
    # print('svt loop done.')
    s_t = tf.cast(s_t,dtype=mat.dtype)
    # print('SVT')
    return s_t, u, v

def shrink(mat, tao, mode='normal'):
    """
    The shrinkage operator, the core of the algorithm. Detial refers in the Paper.

    Parameters
    ----------
    mat:
    The matrix.

    tao:
    The threshold value.It`s a constant given in initializtion.

    mode:str
    'normal' means normal shrinkage operator for matrix B,C.
    'complicated' means a little complicated shrinkage operator for matrix A because of
    the initial defination and constraint.

    Returns
    -------
    The result of shrinkage operator is also a matrix.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3, 4, 5])
    >>> mat1 = tf.random_normal((3, 4))
    >>> tao = tf.constant(0.0)
    >>> tmp_normal = shrink(mat1,tao,mode='normal')
    >>> tmp_complicated = shrink(mat1,tao,mode='complicated')
    >>> tf.Session().run(tmp_normal)
    >>> tf.Session().run(tmp_complicated)
    """
    s, u, v = SVT(mat, tao)
    shape = tf.shape(mat)
    u_shape = tf.shape(u)
    v_shape = tf.shape(v)
    sm_z = tf.zeros((u_shape[0], v_shape[0]), dtype=mat.dtype)
    l = (s.get_shape().as_list())[0]
    indices = []
    values = []
    # print('shrink loop')
    for i in range(l):
        indices.append([i, i])
        values.append(s[i])
    indices = tf.cast(indices, dtype=tf.int64)
    values = tf.cast(values, dtype=s.dtype)
    shape_t = tf.cast(shape, dtype=tf.int64)
    # print('shrink s2d start')
    delta = tf.SparseTensor(indices, values, shape_t)
    sm = tf.sparse_tensor_to_dense(delta, validate_indices=False) + sm_z
    # print('shrink s2d done.')
    if mode == 'normal':
        print('shrink normal')
        return tf.matmul(tf.matmul(u, sm), v)

    if mode == 'complicated':
        vecr1 = tf.ones((shape[0], 1))
        vecr2 = tf.ones((shape[1], 1))
        delta_num = tf.div(tf.matmul(tf.matmul(tf.transpose(vecr1), mat), vecr2), tf.sqrt(tf.cast(shape[0]*shape[1],
                                                                                      dtype=mat.dtype)), name='delta_num')
        tmp1 = tf.matmul(tf.matmul(u, sm), v)
        tmp2 = (tf.maximum(tf.cast(0, dtype=mat.dtype), delta_num-tao)
                +tf.minimum(tf.cast(0, dtype=mat.dtype), delta_num+tao))*tf.ones(shape)/tf.sqrt(tf.cast(shape[0]*shape[1],
                                                                                                      dtype=mat.dtype))
        print('shrink complicated')
        return tmp1+tmp2


def shrinkageBorC(X_hat, tao, r):#no use
    sum = 0
    s = r + 1
    U, S, V = tf.svd(centralization(X_hat))
    '''
    while True:
        if (s + 5 < len(S)):
            s = s + 5
            if (S[s-5] <= tao):
                break
        else:
            if (S[s] <= tao):
                break
            s = s + 1
            if(s>=len(S)):

    for j in range(s-5, s):
        if(S[j] > tao):
            r = j
                 break
    '''
    for i in range(s, tf.size(S).eval()):
        if(S[i]<tao):
            r = i-1
            break


    for j in range(r):
        shape1 = U.get_shape()
        shape2 = V.get_shape()
        m = tf.matmul(tf.reshape(U[j,:],(shape1[0], 1)), tf.reshape(tf.transpose(V[j, :]), (1,shape2[0])))
        sum = sum +(S[j]-tao)*m
    X = sum
    return X, r


def shrinkageA(X_hat, tao, r):# no use
    Xhat_shape=X_hat.get_shape()
    X, r = shrinkageBorC(X_hat, tao, r)
    # delta = np.inner(np.ones(X_hat.shape), X_hat) # elementwise sum of X_hat
    delta = tf.reduce_sum(X_hat)
    gamma = (tf.maximum(0, delta-tao)+tf.minimum(0, delta+tao))/(Xhat_shape[0]*Xhat_shape[1])
    ones_vec1 = tf.reshape(tf.ones(Xhat_shape[0]), (Xhat_shape[0], 1))
    ones_vec2 = tf.reshape(tf.ones(Xhat_shape[1]), (1, Xhat_shape[1]))
    result = X + gamma*tf.matmul(ones_vec1, ones_vec2)

    return result, r






