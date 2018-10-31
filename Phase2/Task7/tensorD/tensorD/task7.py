# import necessary packages
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.cp import CP_ALS
import tensorD.demo.DataGenerator as dg
import tensorflow as tf
import numpy as np


data = np.load("/home/riya/mwd-phase2/tensor.npy")
data_tf = tf.convert_to_tensor(data)
  # features = data["features"]
  # labels = data["labels"]
# use synthetic_data_cp to generate a random tensor with shape of 40x40x40
# X = sdg.synthetic_data_cp([, 20000, 20000], 10)
data_provider = Provider()
data_provider.full_tensor = lambda: data_tf
env = Environment(data_provider, summary_path='/tmp/cp_demo_' + '30')
cp = CP_ALS(env)
# set rank=10 for decomposition
args = CP_ALS.CP_Args(rank=10, validation_internal=1)
# build decomposition model with arguments
cp.build_model(args)
# train decomposition model, set the max iteration as 100
cp.train(100)
# obtain factor matrices from trained model
factor_matrices = cp.factors
for matrix in factor_matrices:
    print(matrix)
# obtain scaling vector from trained model
lambdas = cp.lambdas
print(lambdas)