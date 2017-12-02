# @time    : 2017/12/2 13:32
# @Author  : chiew
# @File    : KNN.py

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_Data/data/", one_hot=True)

xtr, ytr = mnist.train.next_batch(5000)
xte, xte = mnist.test.next_batch(200)