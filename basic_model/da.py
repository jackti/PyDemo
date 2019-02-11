#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  da
    Author:     tigong
    Date:       19-2-2
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class DA(object):
    def __init__(self, inpt, n_visable=784, n_hidden=500, W=None, bhid=None, bvis=None, activation=tf.nn.sigmoid):
        self.n_visiable = n_visable
        self.n_hidden = n_hidden

        if W is None:
            bound = -4 * np.sqrt(6.0 / (self.n_hidden + self.n_visiable))
            W = tf.Variable(tf.random_uniform([self.n_visiable, self.n_hidden], maxval=bound, minval=-bound),
                            dtype=tf.float32)

        if bhid is None:
            bhid = tf.Variable(tf.zeros([n_hidden, ]), dtype=tf.float32)

        if bvis is None:
            bvis = tf.Variable(tf.zeros([n_visable, ]), dtype=tf.float32)

        self.W = W
        self.b = bhid

        self.b_prime = bvis
        self.W_prime = tf.transpose(self.W)

        self.input = inpt

        self.params = [self.W, self.b, self.b_prime]

        self.activation = activation

    def get_decode_values(self, encode_input):
        return self.activation(tf.matmul(encode_input, self.W_prime) + self.b_prime)

    def get_encode_values(self, inpt):
        return self.activation(tf.matmul(inpt, self.W) + self.b)

    def get_corrupted_input(self, inpt, corruption_level):
        input_shape = tf.shape(inpt)
        probs = tf.tile(tf.log([[corruption_level, 1 - corruption_level]]), multiples=[input_shape[0], 1])
        return tf.multiply(tf.cast(tf.multinomial(probs, num_samples=input_shape[1]), dtype=tf.float32), inpt)

    def get_cost(self, corruption_level=0.3):
        corrupted_input = self.get_corrupted_input(self.input, corruption_level)
        encoder_output = self.get_encode_values(corrupted_input)
        decoder_output = self.get_decode_values(encoder_output)

        cross_entropy = tf.multiply(self.input, tf.log(decoder_output)) \
                        + tf.multiply(1 - self.input, tf.log(1 - decoder_output))

        cost = -tf.reduce_mean(cross_entropy)

        return cost


train_steps = 10
display_step = 1
batch_size = 100

if __name__ == '__main__':
    mnist = input_data.read_data_sets("../data/", one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])

    tf.set_random_seed(seed=9999)

    da = DA(x, n_visable=784, n_hidden=500)

    corruption_level = 0.0
    learing_rate = 0.001
    cost = da.get_cost(corruption_level)
    params = da.params

    train_op = tf.train.AdamOptimizer(learing_rate).minimize(cost, var_list=params)

    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        for epoch in range(train_steps):
            avg_cost = 0.0

            batch_num = int(mnist.train.num_examples / batch_size)

            for i in range(batch_num):
                x_batch, y_batch = mnist.train.next_batch(batch_size)

                _, c = sess.run([train_op, cost], feed_dict={x: x_batch})

                avg_cost += c / batch_num

            if epoch % display_step == 0:
                print("Epoch {0} cost : {1}".format(epoch, avg_cost))
                # print("Epoch {0} cost: {1}".format(epoch, avg_cost))
