import numpy as np
import tensorflow as tf
from tensorD.base import pitf_ops_numpy
from tensorD.loss import *
from numpy.random import rand
from .factorization import Model, BaseFact
from .env import Environment


class PITF_np(BaseFact):
    class PITF_np_Args(object):
        def __init__(self,rank,delt,tao,sample_num,validation_internal=-1,verbose=False,steps=1000):
            self.rank = rank
            self.delt = delt
            self.tao = tao
            self.sample_num = sample_num
            self.steps= steps
            self.validation_internal = validation_internal
            self.verbose = verbose

    def __init__(self, env:Environment):
        self._env = env
        self._args = None
        self._model = None
        self._full_tensor = None
        self._sample_vector = None
        self._para =None
        self._is_train_finish = None
        self._init_op = None
        self._sample_vector_update_op = None
        self._erate_op = None
        self._loss_op = None

    def full(self):
        return self._full_tensor

    def exact_recovery(self, args,tensor):
        assert isinstance(args, PITF_np.PITF_np_Args)
        sess = self._env.sess
        #tensor = self._env.full_data()
        #shape = tensor.get_shape().as_list()
        shape = np.shape(tensor)
        spn = args.sample_num
        steps = args.steps
        r = args.rank
        delt = args.delt * spn / np.sqrt(shape[0] * shape[1] * shape[2])
        tao = args.tao * np.sqrt(shape[0] * shape[1] * shape[2])

        y = np.zeros(spn)
        tmpe = 0
        # X, Y, Z = 0, 0, 0
        # X = np.zeros((tensor_shape[0],tensor_shape[1]))
        # Y = np.zeros((tensor_shape[1], tensor_shape[2]))
        # Z = np.zeros((tensor_shape[2], tensor_shape[0]))
        Ef = []
        If = []
        Rf = []
        a, b, c = pitf_ops_numpy.sample3D_rule(tensor.shape, spn)
        sample_list = [a, b, c]
        # A, B, C = sample_rule4mat(tensor_shape, r, r, r, sample)
        # delt = 0.9 * sample_number/ np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
        # tao = 10 * np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
        fd = r * (shape[0] + shape[1] - r) + r * (shape[1] + shape[2] - r) + r * (
            shape[2] + shape[0] - r)
        print('degree of freedom:', fd)
        print('delt:', delt)
        print('m/d:', spn / fd)

        f1 = open('np_e2.txt', 'w')
        f2 = open('np_y.txt', 'w')
        f3 = open('np_X.txt', 'w')

        for i in range(steps):

            X = pitf_ops_numpy.shrink(pitf_ops_numpy.adjoint_operator(sample_list, y, shape, spn, 0), tao, mode='complicated')
            Y = pitf_ops_numpy.shrink(pitf_ops_numpy.adjoint_operator(sample_list, y, shape, spn, 1), tao, mode='normal')
            Z = pitf_ops_numpy.shrink(pitf_ops_numpy.adjoint_operator(sample_list, y, shape, spn, 2), tao, mode='normal')

            X = X / np.sqrt(shape[2])
            Y = Y / np.sqrt(shape[0])
            Z = Z / np.sqrt(shape[1])

            e1 = pitf_ops_numpy.Pomega_tensor(sample_list, tensor, shape, spn)
            # e1 = Pomega_Pair(a,b,c, A, B, C, tensor_shape, sample_number)
            e2 = pitf_ops_numpy.Pomega_Pair(sample_list, X, Y, Z, shape, spn)


            e = e1 - e2

            # print('e:', e)
            # print('e1', e1)
            # print('e2', e2)

            erate = np.linalg.norm(e, ord=2) / np.linalg.norm(e1, ord=2)
            # erate = np.linalg.norm(e, ord=2) / np.linalg.norm(e1, ord=2)

            RMSE_t = 0
            for ii in e:
                RMSE_t += ii ** 2
            loss = np.sqrt(RMSE_t / spn)

            Rf.append(loss)
            Ef.append(erate)
            If.append(i)

            '''
            if (erate <= epsilon):
                print('########################################')
                print(erate)
                break
            if (np.abs(tmpe - erate) < 10e-6 and i > 1000):
                break
            '''

            y = y + delt * e
            # print('y:', y)
            tmpe = erate
            print("%d, %s" % (i, e2), file=f1)
            print("%d, %s" % (i, y), file=f2)
            print("%d, %s" % (i, X), file=f3)

            print('step:%d,erate:%.8f,loss:%.8f' % (i, erate, loss))

        # matlist = [X, Y, Z]
        # return matlist
        f1.close()
        f2.close()
        f3.close()
        return y, X, Y, Z, If, Ef, Rf




