#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
    File Name:  scope.py
    Author:     tigong
    Date:       19-2-7
    Description:
-----------------------------------------------
    Change Activity:
-----------------------------------------------
"""
import tensorflow as tf

tf.reset_default_graph()

# with tf.name_scope("name"):
#     var_1 = tf.Variable(initial_value=[1], name="var1")
#     var_2 = tf.Variable(initial_value=[2], name="var2")
#     var_3 = tf.Variable(initial_value=[3], name="var3")
#
#     var_4 = var_1 + var_2
#
# print(var_1.name)
# print(var_2.name)
# print(var_3.name)
# print(var_4.op.name)

# with tf.name_scope("name_scope"):
#     var_1 = tf.Variable(initial_value=[1], name="var1", dtype=tf.float32)
#     var_2 = tf.get_variable(name="var2", shape=[1, ], dtype=tf.float32)
#     var_5 = var_1 + var_2
#
# with tf.variable_scope("variable_scope"):
#     var_3 = tf.Variable(initial_value=[2], name="var3", dtype=tf.float32)
#     var_4 = tf.get_variable(name="var4", shape=[1, ], dtype=tf.float32)
#     var_6 = var_3 + var_4
#
# print(var_1.name)
# print(var_2.name)
# print(var_5.op.name)
#
# print("--------------")
# print(var_3.name)
# print(var_4.name)
# print(var_6.op.name)

# with tf.variable_scope("scope"):
#     var_1 = tf.get_variable('var1', shape=[1, ])
#
# with tf.variable_scope("scope", reuse=True):
#     var_2 = tf.get_variable("var1", shape=[1, ])
#
# print(var_1.name)
# print(var_2.name)

# with tf.variable_scope("variable_scope") as scope:
#     init = tf.constant_initializer(value=2)
#     var3 = tf.get_variable(name="var3", shape=[1], dtype=tf.float32, initializer=init)
#     scope.reuse_variables()
#     var3_reuse = tf.get_variable(name="var3")
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(var3.name)
#     print(var3_reuse.name)


# def test(name=None):
#     with tf.variable_scope(name, default_name="scope", reuse=tf.AUTO_REUSE) as scope:
#         w = tf.get_variable('w', shape=[2, 10])
#
#
# test("scope")
# test("scope")
#
# ws = tf.trainable_variables()
# for w in ws:
#     print(w.name)
