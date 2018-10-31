#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/4 PM9:03
# @Author  : Shiloh Leung
# @Site    : 
# @File    : cp_demo.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.cp import CP_ALS
from tensorD.demo.DataGenerator import *

if __name__ == '__main__':
    print('=========Train=========')
    X = synthetic_data_cp([40, 40, 40], 10)
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/cp_demo_' + '30')
    cp = CP_ALS(env)
    args = CP_ALS.CP_Args(rank=10, validation_internal=1)
    cp.build_model(args)
    cp.train(100)
    factor_matrices = cp.factors
    lambdas = cp.lambdas
    print('Train ends.\n\n\n')
