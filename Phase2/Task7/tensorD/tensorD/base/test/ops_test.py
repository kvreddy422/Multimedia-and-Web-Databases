# Created by ay27 at 17/1/11
import time
import unittest
from functools import reduce
import logging
import numpy as np
import tensorflow as tf
import tensorD.base.ops as ops
from numpy.random import rand

assert_array_equal = np.testing.assert_array_almost_equal

logger = logging.getLogger('TEST')


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # shape of tmp 3x4x2
        self.tmp = [[[1, 13], [4, 16], [7, 19], [10, 22]],
                    [[2, 14], [5, 17], [8, 20], [11, 23]],
                    [[3, 15], [6, 18], [9, 21], [12, 24]]]
        self.np_x = np.array(self.tmp)
        self.tf_x = tf.constant(self.np_x)

    def test_gen_perm(self):
        x = [2, 3, 1, 0]
        res = ops._gen_perm(4, 2)
        assert_array_equal(x, res)

        x = [0]
        res = ops._gen_perm(1, 0)
        assert_array_equal(x, res)

        # Error test
        with self.assertRaises(ValueError):
            x = []
            res = ops._gen_perm(0, 0)

    def test_unfold(self):
        # mode 1: 3x4x2 -> 4x2x3 -> 4x6
        r1 = np.reshape(np.transpose(self.np_x, [1, 2, 0]), [4, 6])
        with tf.Session().as_default():
            r2 = ops.unfold(self.tf_x, 1).eval()
        assert_array_equal(r1, r2)

        # high order
        t = rand(3, 4, 5, 6, 7, 8)
        r1 = np.reshape(np.transpose(t, [3, 5, 4, 2, 1, 0]), [6, 3 * 4 * 5 * 7 * 8])
        with tf.Session().as_default():
            r2 = ops.unfold(tf.constant(t), 3).eval()
        assert_array_equal(r1, r2)

        # low order
        t = rand(1, 2)
        r1 = t.T
        with tf.Session().as_default():
            r2 = ops.unfold(tf.constant(t, 1), 1).eval()
        assert_array_equal(r1, r2)

        t = rand(2)
        with tf.Session().as_default():
            r2 = ops.unfold(tf.constant(t), 0).eval()
        # must take attention that shape (2,) not equal than (2,1)
        assert_array_equal(np.reshape(t, (2, 1)), r2)

    def test_fold(self):
        # mode 2 : 3x4x2 -> 2x4x3 -> 2x12
        mode_2_mat = np.reshape(np.transpose(self.np_x, [2, 1, 0]), [2, 12])
        mode_2_tf = tf.constant(mode_2_mat)
        with tf.Session().as_default():
            res = ops.fold(mode_2_tf, 2, [3, 4, 2]).eval()
        assert_array_equal(self.np_x, res)

        mat = rand(2, 3)
        with tf.Session().as_default():
            res = ops.fold(tf.constant(mat), 1, (3, 2)).eval()
        assert_array_equal(mat.T, res)

        t = rand(1, 2, 3, 4, 5, 6, 7)
        mat = np.einsum('abcdefg->egfdcba', t).reshape(5, 1 * 2 * 3 * 4 * 6 * 7)
        with tf.Session().as_default():
            res = ops.fold(tf.constant(mat), 4, (1, 2, 3, 4, 5, 6, 7)).eval()
        assert_array_equal(t, res)

    def test_t2mat(self):
        np_A = rand(2, 3, 4, 5)
        tf_A = tf.constant(np_A)

        res1 = np.reshape(np.transpose(np_A, [1, 2, 3, 0]), [12, 10])
        with tf.Session().as_default():
            res2 = ops.t2mat(tf_A, [1, 2], [3, 0]).eval()
        assert_array_equal(res1, res2)

        x = rand(2, 3)
        with tf.Session().as_default():
            res = ops.t2mat(tf.constant(x), 0, 1).eval()
        assert_array_equal(x, res)

        x = rand(3)
        with tf.Session().as_default():
            res = ops.t2mat(tf.constant(x), 0, -1).eval()
        assert_array_equal(np.reshape(x, (3, 1)), res)

    def test_ttm(self):
        np_X = rand(3, 4, 5, 6)
        np_A = rand(3, 4)
        np_B = rand(4, 5)
        np_C = rand(5, 6)
        tf_X = tf.constant(np_X)
        tf_A = tf.constant(np_A)
        tf_B = tf.constant(np_B)
        tf_C = tf.constant(np_C)
        np_res1 = np.einsum('wxyz,xb,yc->wbcz', np_X, np_B, np_C)
        np_res2 = np.matmul(np_A, np_B)

        with tf.Session().as_default():
            tf_res1 = ops.ttm(tf_X, [tf_B, tf_C], transpose=True, axis=[1, 2]).eval()
            tf_res2 = ops.ttm(tf_B, tf_A, axis=[0]).eval()
        assert_array_equal(np_res1, tf_res1, 4)
        assert_array_equal(np_res2, tf_res2, 4)

    def test_vectorize(self):
        res1 = np.reshape(self.np_x, -1)

        with tf.Session().as_default():
            res2 = ops.vectorize(self.tf_x).eval()
        assert_array_equal(res1, res2)

        x = rand(2)
        with tf.Session().as_default():
            res = ops.vectorize(tf.constant(x)).eval()
        assert_array_equal(x, res)

    def test_vec_to_tensor(self):
        np_vec = np.reshape(self.np_x, -1)
        tf_vec = tf.constant(np_vec)
        with tf.Session().as_default():
            res = ops.vec_to_tensor(tf_vec, (3, 4, 2)).eval()
        assert_array_equal(self.np_x, res)

    def test_mul(self):
        np_A = rand(2, 3, 4)
        np_B = rand(4, 5, 6)
        np_res = np.einsum('ijk,klm->ijlm', np_A, np_B)

        tf_A = tf.constant(np_A)
        tf_B = tf.constant(np_B)
        with tf.Session().as_default():
            tf_res = ops.mul(tf_A, tf_B, [2], [0]).eval()
        self.assertEqual(len(np_res.shape), 4)
        np.testing.assert_array_almost_equal(np_res, tf_res)

    def test_mul_run_time(self):
        np_A = rand(20, 30, 400)
        np_B = rand(400, 30, 60)

        ##################################################
        # Also test the run time with numpy, tf.einsum, and ops.mul.
        # Result is very interesting, the speed of einsum is equal to
        #  the ops.mul, but slow than numpy.
        ts1 = time.time()
        for _ in range(10):
            np_res = np.einsum('ijk,kjm->im', np_A, np_B)
        ts2 = time.time()

        tf_A = tf.constant(np_A)
        tf_B = tf.constant(np_B)
        with tf.Session().as_default():
            ts3 = time.time()
            for _ in range(10):
                tf_res = tf.einsum('ijk,kjm->im', tf_A, tf_B).eval()
            ts4 = time.time()
            for _ in range(10):
                tf_res = ops.mul(tf_A, tf_B, [2, 1], [0, 1]).eval()
            ts5 = time.time()
        logger.info('np: %f' % (ts2 - ts1))
        logger.info('tf_einsum: %f' % (ts4 - ts3))
        logger.info('tf ops: %f' % (ts5 - ts4))

    def test_inner(self):
        np_A = rand(2, 3, 4)
        np_B = rand(2, 3, 4)
        np_res = np.sum(np.reshape(np_A, -1) * np.reshape(np_B, -1))

        tf_A = tf.constant(np_A)
        tf_B = tf.constant(np_B)
        with tf.Session().as_default():
            tf_res = ops.inner(tf_A, tf_B).eval()
        np.testing.assert_almost_equal(np_res, tf_res)

    def test_hadamard(self):
        num = 4
        nps = [rand(4, 5) for _ in range(num)]
        np_res = reduce(lambda mata, matb: mata * matb, nps)

        tfs = [tf.constant(nps[i]) for i in range(num)]
        with tf.Session().as_default():
            tf_res = ops.hadamard(tfs).eval()

        np.testing.assert_array_equal(np_res.shape, (4, 5))
        np.testing.assert_array_almost_equal(np_res, tf_res)

        np_res = nps[0] * nps[1] * nps[3]
        with tf.Session().as_default():
            tf_res = ops.hadamard(tfs, skip_matrices_index=2).eval()
        assert_array_equal(np_res, tf_res)

    def test_kron(self):
        np_A = rand(3, 4)
        np_B = rand(5, 6)
        np_res = np.kron(np_A, np_B)

        tf_A = tf.constant(np_A)
        tf_B = tf.constant(np_B)
        with tf.Session().as_default():
            tf_res = ops.kron([tf_A, tf_B]).eval()
        np.testing.assert_array_equal(np_res.shape, [15, 24])
        np.testing.assert_array_almost_equal(np_res, tf_res)

    def test_khatri(self):
        np_A = rand(3, 4)
        np_B = rand(5, 4)
        np_C = rand(6, 4)
        np_res = np.einsum('az,bz,cz->abcz', np_A, np_B, np_C).reshape((90, 4))

        tf_A = tf.constant(np_A)
        tf_B = tf.constant(np_B)
        tf_C = tf.constant(np_C)
        with tf.Session().as_default():
            tf_res = ops.khatri([tf_A, tf_B, tf_C]).eval()
        np.testing.assert_array_almost_equal(np_res, tf_res)

    def test_max_single_value_mul(self):
        np_X = np.array([[2, 0, 1], [-1, 1, 0], [-3, 3, 0]])
        np_matrices = [np_X for _ in range(3)]
        np_res = np.prod([max(np.linalg.svd(mat, compute_uv=0)) for mat in np_matrices])

        tf_matrices = [tf.constant(np_X, dtype=tf.float64) for _ in range(3)]
        with tf.Session().as_default():
            tf_res = ops.max_single_value_mul(tf_matrices).eval()
        np.testing.assert_array_almost_equal(np_res, tf_res)

    def test_xcb(self):
        X = rand(70, 3000)
        np_B = rand(50, 40)
        np_C = rand(60, 40)

        tfB = tf.constant(np_B, tf.float64)
        tfC = tf.constant(np_C, tf.float64)
        X = tf.constant(X)

        with tf.Session().as_default():
            t1 = time.time()
            res = ops.xcb(X, tfC, tfB).eval()
            print("t1 = ", time.time() - t1)
            t2 = time.time()
            res1 = (tf.matmul(X, ops.khatri([tfC, tfB]))).eval()
            print("t2 = ", time.time() - t2)
        np.testing.assert_array_almost_equal(res, res1)


if __name__ == '__main__':
    unittest.main()
