#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  fm.py
    Author:     tigong
    Date:       19-2-18
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""
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


def FM(xtrain, ytrain, epochs=1000, learning_rate=0.001,
       k=10, display_step=200, fm=True, seed=0):
    tf.set_random_seed(seed)

    n = xtrain.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, n], name="x")
    y = tf.placeholder(tf.int32, shape=[None, 1], name="y")

    V = tf.get_variable("V", shape=[n, k], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    W = tf.get_variable("W", shape=[n, 2], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable("b", shape=[2], dtype=tf.float32, initializer=tf.zeros_initializer())

    logits = tf.matmul(x, W) + b
    print("logits:", logits)

    if fm:
        inner = tf.square(tf.matmul(x, V)) - tf.matmul(tf.square(x), tf.square(V))
        fm_hat = tf.multiply(0.5, tf.reduce_sum(inner, axis=1, keep_dims=True))

        logits = logits + fm_hat

    y_hat = tf.nn.sigmoid(logits)


    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y, [-1]), logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            batch_loss, _, y_h = sess.run([loss, optimizer, y_hat], feed_dict={x: xtrain, y: ytrain})

            if epochs % display_step == 0:
                print("Train loss is :%.6f" % batch_loss)


if __name__ == '__main__':
    x_train, y_train, field_dict = dataGenerate(path="../data/fm_train.csv")

    FM(x_train, y_train)
