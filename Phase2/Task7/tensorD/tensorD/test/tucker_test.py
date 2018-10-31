#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/2 PM11:36
# @Author  : Shiloh Leung
# @Site    : 
# @File    : tucker_test.py
# @Software: PyCharm Community Edition
import numpy as np
import tensorflow as tf
from numpy.random import rand
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOSVD
from tensorD.factorization.tucker import HOOI



if __name__ == '__main__':
    data_provider = Provider()
    X = np.arange(60).reshape(3, 4, 5)
    data_provider.full_tensor = lambda: X

    print('====HOSVD test====')
    hosvd_env = Environment(data_provider, summary_path='/tmp/tensord')
    hosvd = HOSVD(hosvd_env)
    args = HOSVD.HOSVD_Args(ranks=[2,2,2])
    hosvd.build_model(args)
    hosvd.train()
    print(hosvd.full - X)

    print('\n\n\n====HOOI test====')
    hooi_env = Environment(data_provider, summary_path='/tmp/tensord')
    hooi = HOOI(hooi_env)
    args = hooi.HOOI_Args(ranks=[2, 2, 2], validation_internal=5)
    hooi.build_model(args)
    hooi.train(100)
    print(hooi.full - X)



