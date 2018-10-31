#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/6 PM3:31
# @Author  : Shiloh Leung
# @Site    : 
# @File    : reader_test.py
# @Software: PyCharm Community Edition

from tensorD.dataproc.reader import TensorReader
import time
import tensorflow as tf

if __name__ == '__main__':
    print('csv file:')
    file_path = 'data1.csv'

    treader = TensorReader(file_path)
    start = time.time()
    treader.read()
    end = time.time()
    print('reader time: %.6f s\n' % (end - start))
    with tf.Session() as sess:
        print(sess.run(treader.sparse_data))
        print(sess.run(treader.full_data))
