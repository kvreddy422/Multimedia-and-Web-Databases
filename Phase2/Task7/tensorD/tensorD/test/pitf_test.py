from tensorD.factorization.env import Environment
from tensorD.factorization.pitf_numpy import PITF_np
from tensorD.factorization.tucker import *
from tensorD.dataproc.provider import Provider
#from tensorD.dataproc.reader import TensorReader
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    data_provider = Provider()
    data_provider.full_tensor = lambda: tf.constant(np.random.rand(50, 50, 8)*10, dtype=tf.float32)
    pitf_np_env = Environment(data_provider, summary_path='/tmp/tensord')
    pitf_np = PITF_np(pitf_np_env)
    sess_t = pitf_np_env.sess
    init_op = tf.global_variables_initializer()
    sess_t.run(init_op)
    tensor = pitf_np_env.full_data().eval(session=sess_t)
    args = PITF_np.PITF_np_Args(rank=5, delt=0.8, tao=12, sample_num=100, validation_internal=1, verbose=False, steps=500)
    y, X_t, Y_t, Z_t, Ef_t, If_t, Rf_t = pitf_np.exact_recovery(args, tensor)
    y = tf.convert_to_tensor(y)
    X = tf.convert_to_tensor(X_t)
    Y = tf.convert_to_tensor(Y_t)
    Z = tf.convert_to_tensor(Z_t)
    Ef = tf.convert_to_tensor(Ef_t)
    If = tf.convert_to_tensor(If_t)
    Rf = tf.convert_to_tensor(Rf_t)


