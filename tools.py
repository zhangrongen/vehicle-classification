# !/usr/bin/python2
# Author: jiangtao
#Project: no_tfrecords_using_slim
#File: tools.py
#Ide: PyCharm
# _*_ coding: utf-8 _*_

import tensorflow as tf

def accuracy(logits,labels):
    correct = tf.equal(tf.arg_max(logits,1),tf.arg_max(labels,1))
    correct = tf.cast(correct,tf.float32)
    accuracy = tf.reduce_mean(correct)*100
    return accuracy

def train_model(loss):

    my_global_step = tf.Variable(0, name='global_step', trainable=False)

    boundaries = [1000,2000,3000,4000,5000]

    lr_values = [0.01,0.005,0.0025,0.001,0.00025,0.0001]

    lr_op = tf.train.piecewise_constant(my_global_step,boundaries,lr_values)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_op)

    train_op = optimizer.minimize(loss,my_global_step)

    return train_op
