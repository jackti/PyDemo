#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  graph.py
    Author:     tigong
    Date:       19-2-7
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    c1 = tf.constant([1.0], dtype=tf.float32)

with tf.Graph().as_default() as g2:
    c2 = tf.constant([2.0])

with tf.Session(graph=g1) as sess1:
    print(sess1.run(c1))

with tf.Session(graph=g2) as sess2:
    print(sess2.run(c2))

#
# tf.reset_default_graph()
#
# with tf.variable_scope("scope_1"):
#     var_1 = tf.get_variable("var1", shape=[1, ])
# with tf.variable_scope("scope_2"):
#     var_2 = tf.get_variable("var1", shape=[1, ])
#
# print(var_1.name)
# print(var_2.name)
