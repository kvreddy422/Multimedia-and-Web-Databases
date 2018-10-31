import numpy as np
import tensorflow as tf
from tensorD.base import pitf_ops
from tensorD.loss import *
from numpy.random import rand
from .factorization import Model, BaseFact
from .env import Environment


class PITF(BaseFact):
    class PITF_Args(object):
        def __init__(self, rank, delt, tao, sample_num, validation_internal=-1, verbose=False):
            self.rank = rank
            self.delt = delt
            self.tao = tao
            self.sample_num = sample_num
            self.validation_internal = validation_internal
            self.verbose = verbose

    def __init__(self, env:Environment):
        self._env = env
        self._args = None
        self._model = None
        self._full_tensor = None
        self._sample_vector = None
        self._is_train_finish = None
        self._init_op = None
        self._sample_vector_update_op = None
        self._erate_op = None
        self._loss_op = None
        self._e = None
        self._e1 = None
        self._e2 = None

    def build_model(self, args):
        assert isinstance(args, PITF.PITF_Args)
        tensor = self._env.full_data()
        shape = tensor.get_shape().as_list()
        spn = args.sample_num
        # r = args.rank
        shape_mul = shape[0]*shape[1]*shape[2]
        delt = tf.truediv(tf.cast(args.delt*spn, dtype=tensor.dtype), tf.sqrt(tf.cast(shape_mul, dtype=tensor.dtype)),
                          name='delta')
        tao = tf.multiply(tf.cast(args.tao, dtype=tensor.dtype), tf.sqrt(tf.cast(shape_mul, dtype=tensor.dtype)),
                          name='tao')
        y = tf.Variable(np.zeros(spn), dtype=tensor.dtype, name='y_sample_vector')
        a, b, c = pitf_ops.sample3D_rule(shape, spn)  # sample list
        spl = [a, b, c]
        # # degree pf freedom
        # fd = r*(shape[0]+shape[1]-r)+r*(shape[1]+shape[2]-r) + \
        #      r*(shape[2]+shape[0]-r)

        with tf.name_scope('adjoint_operator') as scope:
            # with tf.control_dependencies([sample_vector_update_op])# cost little long
            ad1 = pitf_ops.adjoint_operator(spl, y, shape, spn, 0)
            ad2 = pitf_ops.adjoint_operator(spl, y, shape, spn, 1)
            ad3 = pitf_ops.adjoint_operator(spl, y, shape, spn, 2)

        with tf.name_scope('shirnk') as scope:
            with tf.control_dependencies([ad1, ad2, ad3]):
                X = tf.div(pitf_ops.shrink(ad1, tao, mode='complicated'), tf.sqrt(tf.cast(shape[2], dtype=y.dtype),
                                                                                  name='shrink_sqrt1'), name='shrink_div1')
                Y = tf.div(pitf_ops.shrink(ad2, tao, mode='normal'), tf.sqrt(tf.cast(shape[0], dtype=y.dtype),
                                                                                  name='shrink_sqrt2'), name='shrink_div2')
                Z = tf.div(pitf_ops.shrink(ad3, tao, mode='normal'), tf.sqrt(tf.cast(shape[1], dtype=y.dtype),
                                                                                  name='shrink_sqrt3'), name='shrink_div3')
                # X = X / tf.sqrt(tf.cast(shape[2], dtype=X.dtype))
                # Y = Y / tf.sqrt(tf.cast(shape[0], dtype=Y.dtype))
                # Z = Z / tf.sqrt(tf.cast(shape[1], dtype=Z.dtype))
        with tf.name_scope('Pomega_tensor') as scope:# cost very long
            e1 = pitf_ops.Pomega_tensor(spl, tensor, spn)

        with tf.name_scope('Pomega_Pair') as scope:# cost very long
            with tf.control_dependencies([X, Y, Z]):
                e2 = pitf_ops.Pomega_Pair(spl, X, Y, Z, shape, spn)

        with tf.name_scope('error_rate') as scope:
            e = tf.subtract(e1, e2, name='e1-e2')
            # e = pitf_ops.Pomega_tensor(spl, tensor,shape,spn)- pitf_ops.Pomega_Pair(spl, X, Y, Z, shape, spn)
            erate_op = tf.truediv(tf.norm(e, ord='euclidean', name='e_norm'), tf.norm(e1, ord='euclidean',
                                                                                      name='e1_norm'), name='erate')

        with tf.name_scope('loss') as scope:
            RMSE = 0
            length = e.get_shape().as_list()
            for i in range(length[0]):
                RMSE += tf.pow(e[i], 2, name='rmse_update')
            loss_op = tf.sqrt(tf.truediv(RMSE, tf.cast(spn, dtype=RMSE.dtype)), name='rmse_norm')

        with tf.name_scope('update_sum_vec') as scope:
            # y_new = tf.add(y, tf.multiply(delt, e, name='lambda'), name='update_y')
            y_new = y + delt*e
            sample_vector_update_op = y.assign(y_new)

        tf.summary.scalar('error rate', erate_op)
        tf.summary.scalar('loss', loss_op)

        init_op = tf.global_variables_initializer()

        self._args = args
        self._erate_op = erate_op
        self._loss_op = loss_op
        self._sample_vector_update_op = sample_vector_update_op
        self._init_op = init_op
        self._e = e
        self._e1 = e1
        self._e2 = e2

    def full(self):
        return self._full_tensor

    def train_finish(self):
        return self._is_train_finish

    def train(self, steps):
        sess = self._env.sess
        args = self._args
        init_op = self._init_op
        sample_vector_update_op = self._sample_vector_update_op
        erate_op = self._erate_op
        loss_op = self._loss_op
        e = self._e
        e1 = self._e1
        e2 = self._e2

        sum_op = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(self._env.summary_path, sess.graph)
        sess.run(init_op)
        print('PITF model initial finish')

        f1 = open('pitf_y.txt', 'w')
        f2 = open('pitf_e2.txt', 'w')

        for step in range(1, steps+1):
            if (step == steps) or (step == 1) or (step % args.validation_internal == 0 and args.validation_internal !=-1):
                # ea, e1a, e2a,
                e2a, loss_v, erate_v, self._sample_vector, sum_msg = sess.run([e2, loss_op, erate_op,
                                                                              sample_vector_update_op, sum_op])
                sum_writer.add_summary(sum_msg, step)
                # print('ea:', ea)
                # print('e1a:', e1a)
                print('y:%s', self._sample_vector, file=f1)
                print('e2a:%s', e2a, file=f2)
                # print('y:', self._sample_vector)
                print('step=%d, erate=%.8f, RMSE=%.8f' % (step, erate_v, loss_v))
            else:
                self._sample_vector = sess.run(sample_vector_update_op)
                print('???')

        print('tarin finish,with RMSE=%.15f error rate=%.15f' % (loss_v, erate_v))
        self._is_train_finish = True
        f1.close()
        f2.close()


