# Created by ay27 at 17/1/21

import tensorflow as tf
import numpy as np
from tensorD.loss import *
from tensorD.base import *
from .factorization import *
from .env import *


class HOSVD(BaseFact):
    class HOSVD_Args(object):
        def __init__(self, ranks: list):
            self.ranks = ranks

    def __init__(self, env: Environment):
        self._env = env
        self._feed_dict = None
        self._model = None
        self._full_tensor = None
        self._factors = None
        self._core = None
        self._args = None
        self._init_op = None
        self._core_op = None
        self._factor_update_op = None
        self._is_train_finish = False

    def build_model(self, args):
        input_data = tf.placeholder(tf.float32, shape=self._env.full_shape())
        self._feed_dict = {input_data: self._env.full_data()}
        order = input_data.get_shape().ndims
        A = []
        for n in range(order):
            _, U, _ = tf.svd(ops.unfold(input_data, n), full_matrices=True, name='svd-%d' % n)
            A.append(U[:, :args.ranks[n]])
        g = ops.ttm(input_data, A, transpose=True)

        P = TTensor(g, A)

        init_op = tf.global_variables_initializer()
        with tf.name_scope('full-tensor') as scope:
            full_op = P.extract()
        with tf.name_scope('loss') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)

        self._args = args
        self._init_op = init_op
        self._full_op = full_op
        self._factor_update_op = A
        self._core_op = g
        self._loss_op = loss_op

    def train(self, steps=None):
        """

        Parameters
        ----------
        steps : Ignore

        Returns
        -------

        """
        self._is_train_finish = False
        sess = self._env.sess

        sess.run(self._init_op, feed_dict=self._feed_dict)
        print('HOSVD model initial finish')

        loss_v, self._full_tensor, self._factors, self._core = sess.run(
            [self._loss_op, self._full_op, self._factor_update_op, self._core_op], feed_dict=self._feed_dict)
        print('HOSVD model train finish, with RMSE = %f' % loss_v)
        self._is_train_finish = True

    def predict(self, *key):
        return self._full_tensor.item(key)

    @property
    def full(self):
        return self._full_tensor

    @property
    def factors(self):
        return self._factors

    @property
    def core(self):
        return self._core

    @property
    def train_finish(self):
        return self._is_train_finish


class HOOI(BaseFact):
    class HOOI_Args(object):
        def __init__(self, ranks: list, validation_internal=-1, verbose=False, tol=1.0e-4):
            self.ranks = ranks
            self.validation_internal = validation_internal
            self.verbose = verbose
            self.tol = tol

    def __init__(self, env: Environment):
        self._env = env
        self._feed_dict = None
        self._full_tensor = None
        self._factors = None
        self._core = None
        self._args = None
        self._init_op = None
        self._core_op = None
        self._factor_update_op = None
        self._full_op = None
        self._loss_op = None
        self._is_train_finish = False

    def build_model(self, args):
        assert isinstance(args, HOOI.HOOI_Args)
        input_data = tf.placeholder(tf.float32, shape=self._env.full_shape())
        self._feed_dict = {input_data: self._env.full_data()}
        shape = input_data.get_shape().as_list()
        order = input_data.get_shape().ndims

        # HOSVD to initialize factors A
        A = [tf.Variable(tf.random_uniform(shape=(shape[ii], args.ranks[ii]), dtype=tf.float32), name='A-%d' % ii) for
             ii in range(order)]

        init_ops = [None for _ in range(order)]
        for mode in range(order):
            with tf.name_scope('HOSVD-A-init-%d' % mode) as scope:
                _, U, _ = tf.svd(ops.unfold(input_data, mode), full_matrices=True, name='svd-%d' % mode)
                init_ops[mode] = A[mode].assign(U[:, :args.ranks[mode]])

        assign_op = [None for _ in range(order)]
        for mode in range(order):
            if mode != 0:
                with tf.control_dependencies([assign_op[mode - 1]]):
                    with tf.name_scope('Y-%d' % mode) as scope:
                        Y = ops.ttm(input_data, A, skip_matrices_index=mode, transpose=True)
            else:
                with tf.name_scope('Y-%d' % mode) as scope:
                    Y = ops.ttm(input_data, A, skip_matrices_index=mode, transpose=True)
            with tf.name_scope('SVD-%d' % mode) as scope:
                _, tmp, _ = tf.svd(ops.unfold(Y, mode))
                assign_op[mode] = A[mode].assign(tmp[:, :args.ranks[mode]])

        with tf.name_scope('core-tensor') as scope:
            g = ops.ttm(input_data, assign_op, transpose=True)

        init_op = tf.group(*init_ops)

        with tf.name_scope('full-tensor') as scope:
            P = TTensor(g, assign_op)
            full_op = P.extract()

        with tf.name_scope('loss') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)

        tf.summary.scalar('loss', loss_op)

        self._args = args
        self._init_op = init_op
        self._full_op = full_op
        self._factor_update_op = assign_op
        self._core_op = g
        self._loss_op = loss_op

    @property
    def full(self):
        return self._full_tensor

    def predict(self, *key):
        return self._full_tensor.item(key)

    @property
    def factors(self):
        return self._factors

    @property
    def core(self):
        return self._core

    @property
    def train_finish(self):
        return self._is_train_finish

    def train(self, steps):
        sess = self._env.sess
        args = self._args

        init_op = self._init_op
        full_op = self._full_op
        factor_update_op = self._factor_update_op
        core_op = self._core_op
        loss_op = self._loss_op
        loss_hist = []

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op, feed_dict=self._feed_dict)
        nstall = 0
        print('HOOI model initial finish')
        for step in range(1, steps + 1):
            if (step == steps) or args.verbose or (step == 1) or (
                                step % args.validation_internal == 0 and args.validation_internal != -1):
                loss_v, self._full_tensor, self._factors, self._core, sum_msg = sess.run(
                    [loss_op, full_op, factor_update_op, core_op, sum_op], feed_dict=self._feed_dict)
                sum_writer.add_summary(sum_msg, step)
                print('step=%d, RMSE=%.15f' % (step, loss_v))
            else:
                self._factors, self._core, loss_v = sess.run([factor_update_op, core_op, loss_op],
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

        print('HOOI model train finish, in %d steps, with RMSE = %.10f' % (step, loss_v))
        self._is_train_finish = True
        return loss_hist
