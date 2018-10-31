# Created by ay27 at 17/1/16
import unittest

import numpy as np
import tensorflow as tf

from tensorD.base.type import DTensor

from numpy.random import rand

assert_array_equal = np.testing.assert_array_almost_equal


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.np_a = rand(3, 4, 5, 6)
        self.np_b = rand(4, 5, 7, 8)
        self.np_c = rand(2, 8, 7, 3)
        tf_a = tf.constant(self.np_a)

        self.tf_a = DTensor(tf_a)
        self.tf_b = DTensor(self.np_b)
        self.tf_c = DTensor(self.np_c)

    def test_d_mul(self):
        with tf.Session().as_default():
            # 3x4x5x6 mul 4x5x7x8 = 3x6x7x8 mul 2x8x7x3 =3x6x2x3
            tf_res = (self.tf_a.mul(self.tf_b, [1, 2], [0, 1]).mul(self.tf_c, [2, 3], [2, 1])).eval()
        np_res = np.einsum('abcd,bcef,gfeh->adgh', self.np_a, self.np_b, self.np_c)
        assert_array_equal(np_res, tf_res)

    def test_d_dot(self):
        with tf.Session().as_default():
            tf_res = (self.tf_a * self.tf_a * self.np_a).eval()
        np_res = self.np_a * self.np_a * self.np_a
        assert_array_equal(np_res, tf_res)

    def test_d_sub(self):
        with tf.Session().as_default():
            tf_res = (self.tf_a - self.tf_a * 0.5).eval()
        np_res = self.np_a - self.np_a * 0.5
        assert_array_equal(tf_res, np_res)

    def test_d_add(self):
        with tf.Session().as_default():
            tf_res = (self.tf_a + self.tf_a * 0.5 + self.np_a * 0.5).eval()
        np_res = self.np_a * 2
        assert_array_equal(tf_res, np_res)

    def test_d_get_item(self):
        sess = tf.Session()
        tf_res = sess.run(self.tf_a[0][0])
        np_res = self.np_a[0][0]
        assert_array_equal(tf_res, np_res)


if __name__ == '__main__':
    unittest.main()
