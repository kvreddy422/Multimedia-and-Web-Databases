#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/15 PM10:52
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ncp.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np
from numpy.random import rand
from tensorD.loss import *
from tensorD.base import *
from .factorization import *
from .env import *


class NCP_BCU(BaseFact):
    class NCP_Args(object):
        def __init__(self, rank, validation_internal=-1, verbose=False, tol=1.0e-4):
            self.rank = rank
            self.validation_internal = validation_internal
            self.verbose = verbose
            self.tol = tol

    def __init__(self, env):
        assert isinstance(env, Environment)
        self._env = env
        self._feed_dict = None
        self._model = None
        self._full_tensor = None
        self._factors = None
        self._lambdas = None
        self._args = None
        self._init_op = None
        self._other_init_op = None
        self._train_op = None
        self._factor_update_op = None
        self._full_op = None
        self._loss_op = None
        self._is_train_finish = False

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
        assert isinstance(args, NCP_BCU.NCP_Args)
        input_data = tf.placeholder(tf.float32, shape=self._env.full_shape())
        self._feed_dict = {input_data: self._env.full_data()}
        input_norm = tf.norm(input_data)
        shape = input_data.get_shape().as_list()
        order = len(shape)

        with tf.name_scope('random-init') as scope:
            # initialize with normally distributed pseudorandom numbers
            A = [tf.Variable(tf.nn.relu(tf.random_uniform(shape=(shape[ii], args.rank), dtype=tf.float32)),
                             name='A-%d' % ii, dtype=tf.float32) for ii in range(order)]
            A_update_op = [None for _ in range(order)]

        Am = [tf.Variable(np.zeros(shape=(shape[ii], args.rank)), dtype=tf.float32, name='Am-%d' % ii) for ii in
              range(order)]
        A0 = [tf.Variable(np.zeros(shape=(shape[ii], args.rank)), dtype=tf.float32, name='A0-%d' % ii) for ii in
              range(order)]
        Am_update_op1 = [None for _ in range(order)]
        Am_update_op2 = [None for _ in range(order)]
        A0_update_op1 = [None for _ in range(order)]

        with tf.name_scope('norm-init') as scope:
            norm_init_op = [None for _ in range(order)]
            Am_init_op = [None for _ in range(order)]
            A0_init_op = [None for _ in range(order)]
            for mode in range(order):
                norm_init_op[mode] = A[mode].assign(
                    A[mode] / tf.norm(A[mode], ord='fro', axis=(0, 1)) * tf.pow(input_norm, 1 / order))
                A0_init_op[mode] = A0[mode].assign(norm_init_op[mode])
                Am_init_op[mode] = Am[mode].assign(norm_init_op[mode])

        t0 = tf.Variable(1.0, dtype=tf.float32, name='t0')
        t = tf.Variable(1.0, dtype=tf.float32, name='t')
        wA = [tf.Variable(1.0, dtype=tf.float32, name='wA-%d' % ii) for ii in range(order)]
        wA_update_op1 = [None for _ in range(order)]
        L = [tf.Variable(1.0, name='Lipschitz-%d' % ii, dtype=tf.float32) for ii in range(order)]
        L0 = [tf.Variable(1.0, name='Lipschitz0-%d' % ii, dtype=tf.float32) for ii in range(order)]
        L_update_op = [None for _ in range(order)]
        L0_update_op = [None for _ in range(order)]

        with tf.name_scope('unfold-all-mode') as scope:
            mats = [ops.unfold(input_data, mode) for mode in range(order)]

        for mode in range(order):
            if mode != 0:
                with tf.control_dependencies([A_update_op[mode - 1]]):
                    AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in
                           range(order)]
                    XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            else:
                AtA = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
                XA = tf.matmul(mats[mode], ops.khatri(A, mode, True), name='XA-%d' % mode)
            with tf.name_scope('V-%d' % mode):
                V = ops.hadamard(AtA, skip_matrices_index=mode)
            L0_update_op[mode] = L0[mode].assign(L[mode])
            with tf.control_dependencies([L0_update_op[mode]]):
                L_update_op[mode] = L[mode].assign(tf.reduce_max(tf.svd(V, compute_uv=False)))
            Gn = tf.subtract(tf.matmul(Am[mode], V), XA, name='G-%d' % mode)
            with tf.name_scope('A-update-%d' % mode) as scope:
                A_update_op[mode] = A[mode].assign(tf.nn.relu(tf.subtract(Am[mode], tf.div(Gn, L_update_op[mode]))))

        with tf.name_scope('full-tensor') as scope:
            P = KTensor(A_update_op)
            full_op = P.extract()
        with tf.name_scope('loss') as scope:
            loss_op = rmse_ignore_zero(input_data, full_op)
        with tf.name_scope('relative-residual') as scope:
            rel_res_op = tf.norm(full_op - input_data) / input_norm
        with tf.name_scope('objective-value') as scope:
            obj_op = 0.5 * tf.square(tf.norm(full_op - input_data))

        with tf.name_scope('t') as scope:
            t_update_op = t.assign((1 + tf.sqrt(1 + 4 * tf.square(t0))) / 2)
        with tf.name_scope('w') as scope:
            w = (t0 - 1) / t

        with tf.name_scope('Am-wA-update') as scope:
            for mode in range(order):
                # if objective is increasing
                Am_update_op2[mode] = Am[mode].assign(A0[mode])
                # if objective is not increasing
                wA_update_op1[mode] = wA[mode].assign(tf.minimum(w, tf.sqrt(L0[mode] / L[mode])))
                Am_update_op1[mode] = Am[mode].assign(A[mode] + wA_update_op1[mode] * (A[mode] - A0[mode]))
                with tf.control_dependencies([Am_update_op1[mode]]):
                    A0_update_op1[mode] = A0[mode].assign(A[mode])

        with tf.name_scope('t0') as scope:
            with tf.control_dependencies([Am_update_op1[order - 1]]):
                t0_update_op1 = t0.assign(t)

        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('relative_residual', rel_res_op)
        tf.summary.scalar('objective-value', obj_op)

        init_op = tf.global_variables_initializer()

        self._args = args
        self._init_op = init_op
        self._other_init_op = tf.group(*norm_init_op, *Am_init_op, *A0_init_op)
        self._train_op = tf.group(*L_update_op, *L0_update_op, t_update_op)
        self._train_op1 = tf.group(*wA_update_op1, *Am_update_op1, *A0_update_op1, t0_update_op1)
        self._train_op2 = tf.group(*Am_update_op2)
        self._factor_update_op = A_update_op
        self._full_op = full_op
        self._loss_op = loss_op
        self._obj_op = obj_op
        self._rel_res_op = rel_res_op

    def train(self, steps):
        self._is_train_finish = False

        sess = self._env.sess
        args = self._args

        init_op = self._init_op
        other_init_op = self._other_init_op
        factor_update_op = self._factor_update_op
        train_op = self._train_op
        train_op1 = self._train_op1
        train_op2 = self._train_op2
        full_op = self._full_op
        loss_op = self._loss_op
        obj_op = self._obj_op
        rel_res_op = self._rel_res_op
        loss_hist = []

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)

        sess.run(init_op, feed_dict=self._feed_dict)
        sess.run(other_init_op, feed_dict=self._feed_dict)
        nstall = 0
        print('Non-Negative CP model initial finish')

        for step in range(1, steps + 1):
            if (step == steps) or (args.verbose) or (step == 1) or (
                                step % args.validation_internal == 0 and args.validation_internal != -1):
                self._factors, self._full_tensor, loss_v, obj, rel_res, sum_msg, _ = sess.run(
                    [factor_update_op, full_op, loss_op, obj_op, rel_res_op, sum_op, train_op],
                    feed_dict=self._feed_dict)
                sum_writer.add_summary(sum_msg, step)
                print('step=%d, RMSE=%.5f' % (step, loss_v))
            else:
                self._factors, loss_v, rel_res, obj, _ = sess.run(
                    [factor_update_op, loss_op, rel_res_op, obj_op, train_op], feed_dict=self._feed_dict)

            if step == 1:
                obj0 = obj + 1
            loss_hist.append(loss_v)

            relerr1 = abs(obj - obj0) / (obj0 + 1)
            relerr2 = rel_res
            crit = relerr1 < args.tol
            if crit:
                nstall = nstall + 1
            else:
                nstall = 0
            if nstall >= 3 or relerr2 < args.tol:
                break

            if obj < obj0:
                sess.run(train_op1)
                obj0 = obj
            else:
                sess.run(train_op2)

        self._lambdas = np.ones(shape=(1, args.rank))

        print('Non-Negative CP model train finish, in %d steps, with RMSE = %.10f' % (step, loss_v))
        self._is_train_finish = True
        return loss_hist
