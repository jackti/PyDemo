#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  broad.py
    Author:     tigong
    Date:       19-2-6
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""

import tensorflow as tf

a = tf.Variable([[3, 5], [2, 1], [8, 3]], dtype=tf.float32)

b = tf.Variable([2, 3], dtype=tf.float32)

d = tf.multiply(a, b)

with tf.Session() as sess:
    sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(tf.shape(b)))
    print("--------------------------")
    print(sess.run(d))
