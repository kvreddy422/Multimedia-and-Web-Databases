import unittest
from functools import reduce
import tensorflow as tf
import numpy as np
import time
import unittest
from functools import reduce
import logging
import tensorD.base.pitf_ops as pitf_ops
from numpy.random import rand
# all tensor dtype is tf.float32

assert_array_equal = np.testing.assert_array_almost_equal

logger = logging.getLogger('TEST')


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass


    def test_generate(self):
        shape = tf.constant([3, 4])
        rank = tf.constant(3)
        U, V = pitf_ops.generate(shape, rank)


    def test_centalizarion(self):
        mat = tf.random_normal((3, 4))
        result = pitf_ops.centralization(mat)


    def test_subspace(self):
        shape = tf.constant([3, 4])
        rank = tf.constant(3)
        result = pitf_ops.subspace(shape, rank, 'A')
        # result = subspace(shape,rank,'B')
        # result = subspace(shape,rank,'C')


    def test_sample_rule4mat(self):
        shape = tf.constant([3, 4, 5])
        ra = tf.constant(3)
        rb = tf.constant(4)
        rc = tf.constant(5)
        a, b, c = pitf_ops.sample_rule4mat(shape, ra, rb, rc)


    def test_sample3D_rule(self):
        shape = tf.constant([3, 4, 5])
        sample_num = 10
        a, b, c = pitf_ops.sample3D_rule(shape, sample_num)


    def test_Pomega_mat(self):
        shape = tf.constant([3, 4, 5])
        sp_num = 10
        a, b, c = pitf_ops.sample3D_rule(shape, sp_num)
        spl = [a, b, c]
        mat1 = tf.random_normal((3, 4))
        mat2 = tf.random_normal((4, 5))
        mat3 = tf.random_normal((5, 3))
        A = pitf_ops.Pomega_mat(spl, mat1, shape, sp_num, dim=0)
        B = pitf_ops.Pomega_mat(spl, mat2, shape, sp_num, dim=1)
        C = pitf_ops.Pomega_mat(spl, mat3, shape, sp_num, dim=2)


    def test_adjoint_operator(self):
        shape = tf.constant([3, 4, 5])
        sp_num = 10
        a, b, c = pitf_ops.sample3D_rule(shape, sp_num)
        spl = [a, b, c]
        sp_vec = tf.random_uniform([sp_num])
        X = pitf_ops.adjoint_operator(spl, sp_vec, shape, sp_num, dim=0)
        Y = pitf_ops.adjoint_operator(spl, sp_vec, shape, sp_num, dim=1)
        Z = pitf_ops.adjoint_operator(spl, sp_vec, shape, sp_num, dim=2)


    def test_Pomega_tensor(self):
        shape = tf.constant([3, 4, 5])
        sp_num = 20
        a, b, c = pitf_ops.sample3D_rule(shape, sp_num)
        spl = [a, b, c]
        tensor = tf.random_normal((3, 4, 5))
        sp_t = pitf_ops.Pomega_tensor(spl, tensor, sp_num)


    def test_Pomega_Pair(self):
        shape = tf.constant([3, 4, 5])
        sp_num = 10
        a, b, c = pitf_ops.sample3D_rule(shape, sp_num)
        spl = [a, b, c]
        mat1 = tf.random_normal((3, 4))
        mat2 = tf.random_normal((4, 5))
        mat3 = tf.random_normal((5, 3))
        shape = tf.constant([3, 4, 5])
        sp_num = 10
        PA = pitf_ops.Pomega_mat(spl, mat1, shape, sp_num, 0)
        PB = pitf_ops.Pomega_mat(spl, mat2, shape, sp_num, 1)
        PC = pitf_ops.Pomega_mat(spl, mat3, shape, sp_num, 2)
        Pomega_Pair = PA + PB + PC

    """
    def test_cone_projection_operator(self):  # 0.12 version doesn`t have norm function.
        xx = tf.random_normal([5])
        tt = tf.constant(1)
        t1, t2 = pitf_ops.cone_projection_operator(xx, tt)
    """

    def test_SVT(self):
        shape = tf.constant([3, 4, 5])
        mat1 = tf.random_normal((3, 4))
        tao = tf.constant(0.0)
        s, u, v = pitf_ops.SVT(mat1, tao)


    def test_shrink(self):  # false
        shape = tf.constant([3, 4, 5])
        mat1 = tf.random_normal((3, 4))
        tao = tf.constant(0.0)
        # print('matrix shape:', mat1.get_shape().as_list())
        tmp_normal = pitf_ops.shrink(mat1, tao, mode='normal')
        tmp_complicated = pitf_ops.shrink(mat1, tao, mode='complicated')


if __name__ == '__main__':
    unittest.main()
