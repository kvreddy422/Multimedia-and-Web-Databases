#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/4 PM9:39
# @Author  : Shiloh Leung
# @Site    : 
# @File    : tucker_demo.py
# @Software: PyCharm Community Edition

from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOOI
from tensorD.demo.DataGenerator import *

if __name__ == '__main__':
    print('=========Train=========')
    X = synthetic_data_tucker([40, 40, 40], [10, 10, 10])
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/tucker_demo_' + '30')
    hooi = HOOI(env)
    args = HOOI.HOOI_Args(ranks=[10, 10, 10], validation_internal=1, tol=1.0e-4)
    hooi.build_model(args)
    hooi.train(50)
    factor_matrices = hooi.factors
    core_tensor = hooi.core
    print('Train ends.\n\n\n')
