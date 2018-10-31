#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/3 PM4:16
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ntucker_demo.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.ntucker import NTUCKER_BCU
from tensorD.demo.DataGenerator import *

if __name__ == '__main__':
    print('=========Train=========')
    X = synthetic_data_tucker([30, 30, 30], [10, 10, 10])
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/ntucker_demo_' + '30')
    ntucker = NTUCKER_BCU(env)
    args = NTUCKER_BCU.NTUCKER_Args(ranks=[10, 10, 10], validation_internal=1)
    ntucker.build_model(args)
    ntucker.train(1000)
    factor_matrices = ntucker.factors
    core_tensor = ntucker.core
    print('Train ends.\n\n\n')
