# Created by ay27 at 17/1/16
import unittest

import numpy as np
import tensorflow as tf

from tensorD.base.type import KTensor

from numpy.random import rand
assert_array_equal = np.testing.assert_array_almost_equal


class MyTestCase(unittest.TestCase):
    def test_extract(self):
        x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
        u,s,v = np.linalg.svd(x, full_matrices=False)

        ktensor = KTensor([u, v.T], s)
        with tf.Session().as_default():
            res = ktensor.extract().eval()
        assert_array_equal(x, res)

    def test_high_order(self):
        a = rand(5,4)
        b = rand(6,4)
        c = rand(7,4)
        d = rand(8,4)
        # lambdas = [2,3,4,5]

        kt = KTensor([a,b,c,d])
        with tf.Session().as_default():
            res = kt.extract().eval()
        assert_array_equal(res.shape, [5,6,7,8])


if __name__ == '__main__':
    unittest.main()
