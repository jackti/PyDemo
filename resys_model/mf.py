#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  mf.py
    Author:     tigong
    Date:       19-2-6
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""
import time

import tensorflow as tf
import numpy as np
import os


class MF(object):
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=0.01,
                 epoch=500, batch_size=128, show_time=False, T=2, display_step=100):
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.sess = sess
        self.epochs = epoch
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.show_time = show_time
        self.T = T
        self.display_step = display_step

        print("MF model")

    def build_graph(self, num_factor=50):
        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder(tf.float32, shape=[None], name='rating')

        self.P = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev=0.01))
        self.Q = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev=0.01))

        self.Bias_U = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        self.Bias_I = tf.Variable(tf.random_normal([self.num_item], dtype=0.01))

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_id)
        user_bias = tf.nn.embedding_lookup(self.Bias_U, self.user_id)
        item_bias = tf.nn.embedding_lookup(self.Bias_I, self.item_id)

        interaction = tf.multiply(user_latent_factor, item_latent_factor)
        self.pred_rating = tf.reduce_sum(interaction, axis=1) + user_bias + item_bias

    def execute(self, train_data, test_data):
        t = train_data.tocoo()
        self.user = t.row.reshape(-1)
        self.item = t.col.reshape(-1)
        self.rating = t.data

        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) + \
                    self.reg_rate * (tf.nn.l2_loss(self.Bias_U) + tf.nn.l2_loss(self.Bias_I) +
                                     tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, train_data):
        self.num_train = len(train_data)
        num_batch = int(self.num_train / self.batch_size)
        idx = np.random.permutation(self.num_train)
        shuffle_user = list(self.user[idx])
        shuffle_item = list(self.item[idx])
        shuffle_rating = list(self.rating[idx])

        for i in range(num_batch):
            start_time = time.time()

            batch_user = shuffle_user[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = shuffle_item[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = shuffle_rating[i * self.batch_size:(i + 1) * self.batch_size]

            self.sess.run([])

            print("hello");
            print("testx")
