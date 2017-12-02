# @time    : 2017/12/2 13:32
# @Author  : chiew
# @File    : KNN.py

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_Data/data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

distance = tf.reduce_sum(
    tf.abs(
        tf.add(
            xtr,
            tf.negative(xte))),
    reduction_indices=1)

pred = tf.argmin(distance, 0)

accuracy = 0.

init = tf.global_variables_initializer()

# Launch graph
with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):

        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print("nn_index", nn_index)
        print("Ytr[nn_index]", Ytr[nn_index])
        print(
            "np.argmax(Ytr[nn_index]):", np.argmax(
                Ytr[nn_index]), "True Class", np.argmax(
                Yte[i]))

        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)

    print("Done!")
    print("Accuracy:", accuracy)
