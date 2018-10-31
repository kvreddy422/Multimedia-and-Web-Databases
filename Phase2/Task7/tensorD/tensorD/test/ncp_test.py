#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/15 PM10:53
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ncp_test.py
# @Software: PyCharm Community Edition
from tensorD.factorization.env import Environment
from tensorD.factorization.ncp import NCP_BCU
from tensorD.dataproc.provider import Provider
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    data_provider = Provider()
    X = np.arange(60).reshape(3, 4, 5)
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/tensord')
    ncp = NCP_BCU(env)
    args = NCP_BCU.NCP_Args(rank=2, validation_internal=5)
    ncp.build_model(args)
    ncp.train(500)

