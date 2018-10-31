#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 PM7:22
# @Author  : Shiloh Leung
# @Site    :
# @File    : cp_ex.py
# @Software: PyCharm Community Edition

from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.ntucker import NTUCKER_BCU
from tensorD.factorization.tucker import HOOI
from tensorD.factorization.cp import CP_ALS
from tensorD.factorization.ncp import NCP_BCU
from tensorD.demo.DataGenerator import *
import sys


def cp_run(N1, N2, N3, gR, dR, time):
    # cp test
    X = synthetic_data_cp([N1, N2, N3], gR, 0)
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/cp_' + str(N1))
    cp = CP_ALS(env)
    args = CP_ALS.CP_Args(rank=dR, validation_internal=200, tol=1.0e-4)
    cp.build_model(args)
    print('CP with %dx%dx%d, gR=%d, dR=%d, time=%d' % (N1, N2, N3, gR, dR, time))
    loss_hist = cp.train(6000)
    scale = str(N1) + '_' + str(gR) + '_' + str(dR)
    out_path = '/root/tensorD_f/data_out_tmp/python_out/cp_' + scale + '_' + str(time) + '.txt'
    with open(out_path, 'w') as out:
        for loss in loss_hist:
            out.write('%.6f\n' % loss)


if __name__ == '__main__':
    cp_run(N1=int(sys.argv[1]),
           N2=int(sys.argv[2]),
           N3=int(sys.argv[3]),
           gR=int(sys.argv[4]),
           dR=int(sys.argv[5]),
           time=int(sys.argv[6]))
