#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/4 PM8:41
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ncp_demo.py
# @Software: PyCharm Community Edition


from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.ncp import NCP_BCU
from tensorD.demo.DataGenerator import *

if __name__ == '__main__':
    print('=========Train=========')
    X = synthetic_data_cp([30, 30, 30], 10)
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/ncp_demo_' + '30')
    ncp = NCP_BCU(env)
    args = NCP_BCU.NCP_Args(rank=10, validation_internal=1)
    ncp.build_model(args)
    ncp.train(100)
    factor_matrices = ncp.factors
    lambdas = ncp.lambdas
    print('Training ends.\n\n\n')
