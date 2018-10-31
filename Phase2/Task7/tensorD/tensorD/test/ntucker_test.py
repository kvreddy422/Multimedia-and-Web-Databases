#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/23 PM4:12
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ntucker_test.py
# @Software: PyCharm Community

import numpy as np
import tensorflow as tf
from numpy.random import rand
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.ntucker import NTUCKER_BCU

if __name__ == '__main__':
    data_provider = Provider()
    X = np.arange(60).reshape(3, 4, 5)
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/tensord')
    ntucker = NTUCKER_BCU(env)
    args = NTUCKER_BCU.NTUCKER_Args(ranks=[2, 2, 2], validation_internal=1)
    ntucker.build_model(args)
    ntucker.train(500)
