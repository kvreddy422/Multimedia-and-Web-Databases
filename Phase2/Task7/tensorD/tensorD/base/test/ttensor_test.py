# Created by ay27 at 17/1/15
import unittest

import numpy as np
import tensorflow as tf

from tensorD.base.type import TTensor

from numpy.random import rand


class MyTestCase(unittest.TestCase):
    def test_extract(self):
        g = rand(2, 3, 4)
        a = rand(5, 2)
        b = rand(6, 3)
        c = rand(7, 4)

        res1 = np.einsum('xyz,ax,by,cz->abc', g, a, b, c)

        tg = tf.constant(g)
        ta = tf.constant(a)
        tb = tf.constant(b)
        tc = tf.constant(c)

        with tf.Session().as_default():
            tt = TTensor(tg, [ta, tb, tc])
            res2 = tt.extract().eval()

        np.testing.assert_array_almost_equal(res1, res2)

if __name__ == '__main__':
    unittest.main()
