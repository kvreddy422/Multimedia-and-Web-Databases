# Created by ay27 at 17/6/6
from tensorD.factorization.env import Environment
from tensorD.factorization.cp import CP_ALS
from tensorD.factorization.tucker import *
from tensorD.dataproc.provider import Provider
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    data_provider = Provider()
    X = np.arange(60).reshape(3, 4, 5)
    data_provider.full_tensor = lambda: X
    env = Environment(data_provider, summary_path='/tmp/tensord')
    cp = CP_ALS(env)
    args = CP_ALS.CP_Args(rank=2, validation_internal=5)
    cp.build_model(args)
    cp.train(500)
    print(cp.full - X)

