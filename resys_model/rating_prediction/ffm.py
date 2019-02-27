#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  ffm.py
    Author:     tigong
    Date:       19-2-27
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def dataGenerate(path="../data/fm_train.csv"):
    df = pd.read_csv(path)
    df = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
    class_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
    continuous_columns = ['Fare']

    train_x = df.drop('Survived', axis=1)
    train_y = df['Survived'].values
    train_x = train_x.fillna('-1')

    le = LabelEncoder()
    oht = OneHotEncoder()

    files_dict = {}
    s = 0
    for idx, column in enumerate(class_columns):
        try:
            train_x[column] = le.fit_transform(train_x[column])
        except Exception as e:
            print("Exception---%s" % column)
            pass

        ont_x = oht.fit_transform(train_x[column].values.reshape(-1, 1)).toarray()

        for i in range(ont_x.shape[1]):
            files_dict[s] = idx
            s += 1

        if idx == 0:
            x_t = ont_x
        else:
            x_t = np.hstack((x_t, ont_x))

    x_t = np.hstack((x_t, train_x[continuous_columns].values.reshape(-1, 1)))
    files_dict[s] = idx + 1

    return x_t, train_y.reshape(-1, 1), files_dict


def FFM(xtrain, ytrain, field_dict, train_steps=1000, learning_rate=0.001, K=10,
        display_information=100, ffm=True, seed=0):
    tf.set_random_seed(seed=seed)

    n = xtrain.shape[1]

    f = sorted(field_dict.items(), key=lambda x: x[1], reverse=True)[0][1]

    x = tf.placeholder(tf.float32, shape=[None, n], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="y")

    V = tf.get_variable("V", shape=[f + 1, n, K], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.3))
    W = tf.get_variable("W", shape=[n, 1], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.3))
    b = tf.get_variable("b", shape=[1, 1], dtype=tf.float32, initializer=tf.zeros_initializer())

    logits = tf.matmul(x, W) + b

    if ffm:
        fm_hat = tf.constant(0.0, dtype=tf.float32)

        for i in range(n):
            for j in range(i + 1, n):
                fm_hat += tf.multiply(
                    tf.reduce_mean(tf.multiply(V[field_dict[j], i], V[field_dict[i], j])),
                    tf.reshape(tf.multiply(x[:, i], x[:, j]), [-1, 1])
                )

        logits = logits + fm_hat

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    y_hat = tf.nn.sigmoid(logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for i in range(train_steps):
            batch_loss, _, batch_y = sess.run([loss, optimizer, y_hat], feed_dict={x: xtrain, y: ytrain})

            if i % display_information == 0:
                print("Train step=%4d  Train loss=%.6f" % (i, batch_loss))


if __name__ == '__main__':
    x_train, y_train, field_dict = dataGenerate(path="../data/fm_train.csv")

    FFM(x_train, y_train, field_dict)
