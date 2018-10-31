# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 PM11:01
# @Author  : Shiloh Leung
# @Site    :
# @File    : tucker_ex.py
# @Software: PyCharm Community Edition

from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.tucker import HOOI
from tensorD.demo.DataGenerator import *
import sys


def tucker_run(N1, N2, N3, gR, dR, time):
    # tucker
    X = synthetic_data_tucker([N1, N2, N3], [gR, gR, gR])
    data_provider = Provider()
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/tucker_' + str(N1))
    hooi = HOOI(env)
    args = HOOI.HOOI_Args(ranks=[dR, dR, dR], validation_internal=200)
    hooi.build_model(args)
    print('\n\nTucker with %dx%dx%d, gR=%d, dR=%d, time=%d' % (N1, N2, N3, gR, dR, time))
    loss_hist = hooi.train(6000)
    scale = str(N1) + '_' + str(gR) + '_' + str(dR)
    out_path = '/root/tensorD_f/data_out_tmp/python_out/tucker_' + scale + '_' + str(time) + '.txt'
    with open(out_path, 'w') as out:
        for loss in loss_hist:
            out.write('%.6f\n' % loss)


if __name__ == '__main__':
    tucker_run(N1=int(sys.argv[1]),
               N2=int(sys.argv[2]),
               N3=int(sys.argv[3]),
               gR=int(sys.argv[4]),
               dR=int(sys.argv[5]),
               time=int(sys.argv[6]))
