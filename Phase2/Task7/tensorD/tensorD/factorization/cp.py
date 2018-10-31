# Created by ay27 at 17/1/13
import numpy as np
import tensorflow as tf
from tensorD.base import *
from tensorD.loss import *
from numpy.random import rand
from .factorization import Model, BaseFact
from .env import Environment


class CP_ALS(BaseFact):
    class CP_Args(object):
        def __init__(self,
                     rank,
                     tol=1.0e-4,
                     validation_internal=-1,
                     get_lambda=False,
                     get_rmse=False,
                     verbose=False):
            self.rank = rank
            self.tol = tol
            self.validation_internal = validation_internal
            self.get_lambda = get_lambda
            self.get_rmse = get_rmse
            self.verbose = verbose

    def __init__(self, env):
        assert isinstance(env, Environment)
        self._env = env
        self._feed_dict = None
        self._model = None
        self._full_tensor = None
        self._factors = None
        self._lambdas = None
        self._is_train_finish = False
        self._args = None
        self._init_op = None
        self._norm_input_data = None
        self._lambda_op = None
        self._full_op = None
        self._factor_update_op = None
        self._fit_op_zero = None
        self._fit_op_not_zero = None
        self._loss_op = None

    def predict(self, *key):
        if not self._full_tensor:
            raise TensorErr('improper stage to call predict before the model is trained')
        return self._full_tensor.item(key)

    @property
    def full(self):
        return self._full_tensor

    @property
    def train_finish(self):
        return self._is_train_finish

    @property
    def factors(self):
        return self._factors

    @property
    def lambdas(self):
        return self._lambdas

    def build_model(self, args):
        assert isinstance(args, CP_ALS.CP_Args)
        input_data = tf.placeholder(tf.float32, shape=self._env.full_shape())
        self._feed_dict = {input_data: self._env.full_data()}
        shape = input_data.get_shape().as_list()
        order = len(shape)

        with tf.name_scope('random-init') as scope:
            A = [tf.Variable(tf.random_uniform(shape=(shape[ii], args.rank), dtype=tf.float32), name='A-%d' % ii) for ii
                 in range(order)]
        with tf.name_scope('unfold-all-mode') as scope:
            mats = [ops.unfold(input_data, mode) for mode in range(order)]
            assign_op = [None for _ in range(order)]

        for mode in range(order):
            if mode != 0:
                with tf.control_dependencies([assign_op[mode - 1]]):
                    AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in
                           range(order)]
                    XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            else:
                AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
                XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)

            V = ops.hadamard(AtA, skip_matrices_index=mode)
            non_norm_A = tf.matmul(XA, tf.py_func(np.linalg.pinv, [V], tf.float32, name='pinvV-%d' % mode),
                                   name='XApinvV-%d' % mode)
            with tf.name_scope('max-norm-%d' % mode) as scope:
                lambda_op = tf.reduce_max(tf.reshape(non_norm_A, shape=(shape[mode], args.rank)), axis=0)
                assign_op[mode] = A[mode].assign(tf.div(non_norm_A, lambda_op))

        with tf.name_scope('full-tensor') as scope:
            P = KTensor(assign_op, lambda_op)
            full_op = P.extract()

        with tf.name_scope('loss') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)

        tf.summary.scalar('loss', loss_op)

        init_op = tf.global_variables_initializer()

        self._args = args
        self._init_op = init_op
        self._lambda_op = lambda_op
        self._full_op = full_op
        self._factor_update_op = assign_op
        self._loss_op = loss_op

    def train(self, steps):
        self._is_train_finish = False

        sess = self._env.sess
        args = self._args

        init_op = self._init_op

        lambda_op = self._lambda_op
        full_op = self._full_op
        factor_update_op = self._factor_update_op
        loss_op = self._loss_op
        loss_hist = []

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op, feed_dict=self._feed_dict)
        nstall = 0
        print('CP model initial finish')

        for step in range(1, steps + 1):
            if (step == steps) or (args.verbose) or (step == 1) or (
                        step % args.validation_internal == 0 and args.validation_internal != -1):
                self._factors, self._lambdas, self._full_tensor, loss_v, sum_msg = sess.run(
                    [factor_update_op, lambda_op, full_op, loss_op, sum_op], feed_dict=self._feed_dict)
                sum_writer.add_summary(sum_msg, step)
                print('step=%d, RMSE=%f' % (step, loss_v))
            else:
                self._factors, self._lambdas, loss_v = sess.run([factor_update_op, lambda_op, loss_op],
                                                                feed_dict=self._feed_dict)
            loss_hist.append(loss_v)
            if step == 1:
                loss_v0 = loss_v + 1

            relerr1 = abs(loss_v - loss_v0) / (loss_v0 + 1)
            relerr2 = abs(loss_v - loss_v0)
            crit = relerr1 < args.tol
            if crit:
                nstall = nstall + 1
            else:
                nstall = 0
            if nstall >= 3 or relerr2 < args.tol:
                break
            loss_v0 = loss_v

        print('CP model train finish, in %d steps, with RMSE = %.10f' % (step, loss_v))
        self._is_train_finish = True
        return loss_hist
