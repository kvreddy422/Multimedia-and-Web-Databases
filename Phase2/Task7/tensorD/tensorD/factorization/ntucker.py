#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/16 AM11:48
# @Author  : Shiloh Leung
# @Site    : 
# @File    : ntucker.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import numpy as np
from tensorD.loss import *
from tensorD.base import *
from numpy.random import rand
from functools import reduce
from .factorization import BaseFact
from .env import Environment


class NTUCKER_BCU(BaseFact):
    class NTUCKER_Args(object):
        def __init__(self, ranks: list, validation_internal=-1, verbose=False, tol=1.0e-4):
            self.ranks = ranks
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
        self._core = None
        self._args = None
        self._init_op = None
        self._core_init = None
        self._core_op = None
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
    def factors(self):
        return self._factors

    @property
    def core(self):
        return self._core

    @property
    def train_finish(self):
        return self._is_train_finish

    def build_model(self, args):
        assert isinstance(args, NTUCKER_BCU.NTUCKER_Args)
        input_data = tf.placeholder(tf.float32, shape=self._env.full_shape())
        self._feed_dict = {input_data: self._env.full_data()}
        input_norm = tf.norm(input_data)
        shape = input_data.get_shape().as_list()
        order = input_data.get_shape().ndims

        with tf.name_scope('random-init') as scope:
            # initialize with normally distributed pseudorandom numbers
            A = [tf.Variable(tf.nn.relu(tf.random_uniform(shape=(shape[ii], args.ranks[ii]), dtype=tf.float32)),
                             name='A-%d' % ii, dtype=tf.float32) for ii in range(order)]

            A_update_op = [None for _ in range(order)]

        Am = [tf.Variable(np.zeros(shape=(shape[ii], args.ranks[ii])), dtype=tf.float32, name='Am-%d' % ii) for ii in
              range(order)]
        Am_update_op1 = [None for _ in range(order)]
        Am_update_op2 = [None for _ in range(order)]
        A0 = [tf.Variable(np.zeros(shape=(shape[ii], args.ranks[ii])), dtype=tf.float32, name='A0-%d' % ii) for ii in
              range(order)]
        A0_update_op1 = [None for _ in range(order)]

        with tf.name_scope('norm-init') as scope:
            norm_init_op = [None for _ in range(order)]
            Am_init_op = [None for _ in range(order)]
            A0_init_op = [None for _ in range(order)]
            for mode in range(order):
                norm_init_op[mode] = A[mode].assign(
                    A[mode] / tf.norm(A[mode], ord='fro', axis=(0, 1)) * tf.pow(input_norm, 1 / (order + 1)))
                A0_init_op[mode] = A0[mode].assign(norm_init_op[mode])
                Am_init_op[mode] = Am[mode].assign(norm_init_op[mode])

        # initialize with normally distributed pseudorandom numbers
        g = tf.Variable(tf.nn.relu(tf.random_uniform(shape=args.ranks, dtype=tf.float32)), name='core-tensor')
        with tf.name_scope('core-init') as scope:
            g_norm_init = g.assign(g / tf.norm(g) * tf.pow(input_norm, 1 / (order + 1)))
        g0 = tf.Variable(np.zeros(shape=args.ranks), dtype=tf.float32, name='core_0')
        with tf.name_scope('core_0-init') as scope:
            g0_init_op = g0.assign(g_norm_init)
        gm = tf.Variable(np.zeros(shape=args.ranks), dtype=tf.float32, name='core_m')
        with tf.name_scope('core_m-init') as scope:
            gm_init_op = gm.assign(g_norm_init)

        t0 = tf.Variable(1.0, dtype=tf.float32, name='t0')
        t = tf.Variable(1.0, dtype=tf.float32, name='t')
        wA = [tf.Variable(1.0, dtype=tf.float32, name='wA-%d' % ii) for ii in range(order + 1)]
        wA_update_op1 = [None for _ in range(order + 1)]

        L = [tf.Variable(1.0, name='L-%d' % ii, dtype=tf.float32) for ii in range(order + 1)]
        L0 = [tf.Variable(1.0, name='L0-%d' % ii, dtype=tf.float32) for ii in range(order + 1)]
        L_update_op = [None for _ in range(order + 1)]
        L0_update_op = [None for _ in range(order + 1)]

        Bsq = tf.Variable(np.zeros(shape=args.ranks), dtype=tf.float32, name='Bsq')
        Grad_g = tf.Variable(np.zeros(shape=args.ranks), dtype=tf.float32, name='Grad-core')

        with tf.name_scope('unfold-all-mode') as scope:
            mats = [ops.unfold(input_data, mode) for mode in range(order)]

        # update core tensor g
        AtA_g = [tf.matmul(A[ii], A[ii], transpose_a=True, name='AtA-%d-%d' % (mode, ii)) for ii in range(order)]
        with tf.name_scope('core-update') as scope:
            L0_update_op[order] = L0[order].assign(L[order])
            with tf.control_dependencies([L0_update_op[order]]):
                L_update_op[order] = L[order].assign(ops.max_single_value_mul(AtA_g))
            Bsq_update_op_g = Bsq.assign(ops.ttm(gm, AtA_g))
            Grad_g_update_op = Grad_g.assign(Bsq_update_op_g - ops.ttm(input_data, A, transpose=True))
            g_update_op = g.assign(tf.nn.relu(gm - Grad_g_update_op / L_update_op[order]))

        # update factor matrices A
        for mode in range(order):
            with tf.name_scope('B-%d' % mode) as scope:
                if mode != 0:
                    with tf.control_dependencies([A_update_op[mode - 1]]):
                        B = ops.unfold(ops.ttm(g, A, skip_matrices_index=mode), mode)
                else:
                    B = ops.unfold(ops.ttm(g, A, skip_matrices_index=mode), mode)
            XB = tf.matmul(mats[mode], B, transpose_b=True, name='XB-%d' % mode)
            Bsq_A = tf.matmul(B, B, transpose_b=True, name='Bsq_A-%d' % mode)
            GradA = tf.subtract(tf.matmul(Am[mode], Bsq_A), XB, name='GradA-%d' % mode)
            with tf.name_scope('A-update-%d' % mode) as scope:
                L0_update_op[mode] = L0[mode].assign(L[mode])
                with tf.control_dependencies([L0_update_op[mode]]):
                    L_update_op[mode] = L[mode].assign(tf.reduce_max(tf.svd(Bsq_A, compute_uv=False)))
                A_update_op[mode] = A[mode].assign(tf.nn.relu(tf.subtract(Am[mode], tf.div(GradA, L_update_op[mode]))))

        with tf.name_scope('full-tensor') as scope:
            P = TTensor(g, A_update_op)
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

        for mode in range(order):
            # if objective is increasing
            Am_update_op2[mode] = Am[mode].assign(A0[mode])
            gm_update_op2 = gm.assign(g0)
            # if objective is not increasing
            wA_update_op1[mode] = wA[mode].assign(tf.minimum(w, tf.sqrt(L0[mode] / L[mode])))
            Am_update_op1[mode] = Am[mode].assign(A[mode] + wA_update_op1[mode] * (A[mode] - A0[mode]))
            with tf.control_dependencies([Am_update_op1[mode]]):
                A0_update_op1[mode] = A0[mode].assign(A[mode])
        wA_update_op1[order] = wA[order].assign(tf.minimum(w, tf.sqrt(L0[order] / L[order])))

        gm_update_op1 = gm.assign(g + wA_update_op1[order] * (g - g0))
        with tf.control_dependencies([gm_update_op1]):
            g0_update_op1 = g0.assign(g)
        with tf.control_dependencies([Am_update_op1[order - 1]]):
            t0_update_op1 = t0.assign(t)

        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('relative_residual', rel_res_op)
        tf.summary.scalar('objective-value', obj_op)

        init_op = tf.global_variables_initializer()

        self._args = args
        self._init_op = init_op
        self._other_init_op = tf.group(*norm_init_op, *A0_init_op, *Am_init_op, g_norm_init, g0_init_op, gm_init_op)
        self._core_train_op = tf.group(L0_update_op[order], L_update_op[order], Bsq_update_op_g)
        self._core_update_op = g_update_op
        self._factor_update_op = A_update_op
        self._train_op = tf.group(*L0_update_op[0:order - 1], *L_update_op[0:order - 1], t_update_op)
        self._train_op1 = tf.group(*wA_update_op1, *Am_update_op1, *A0_update_op1, gm_update_op1, g0_update_op1,
                                   t0_update_op1)
        self._train_op2 = tf.group(*Am_update_op2, gm_update_op2)
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
        core_train_op = self._core_train_op
        core_update_op = self._core_update_op
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
        print('Non-Negative Tucker model initial finish')

        for step in range(1, steps + 1):
            self._core, _ = sess.run([core_update_op, core_train_op], feed_dict=self._feed_dict)
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
            loss_hist.append(loss_v)
            if step == 1:
                obj0 = obj + 1

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

        print('Non-Negative Tucker model train finish, in %d steps, with RMSE = %.10f' % (step, loss_v))
        self._is_train_finish = True
        return loss_hist
