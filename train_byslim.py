# !/usr/bin/python2
# Author: jiangtao
#Project: classify
#File: train_byslim.py
#Ide: PyCharm
# _*_ coding: utf-8 _*_

import tensorflow as tf
import tools
import numpy as np
import os
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import inception,vgg,resnet_v2
from gen_train_batch import get_file,get_batch
from tensorflow.python.ops import control_flow_ops

model_path = '/home/jiangtao/model/5_class_vgg/'
train_log_dir ='/home/jiangtao/logs/5_class_vgg/'


file_dir = '/media/jiangtao/新加卷/datasets/DANdatatrain/'
image_list, label_list = get_file(file_dir)


def train():

    with tf.Graph().as_default():

        with tf.name_scope('input'):

            tra_image_batch, tra_label_batch = get_batch(image_list, label_list)



        x = tf.placeholder(tf.float32, shape=[32, 224, 224, 3])
        y_gt = tf.placeholder(tf.int16, shape=[32, 5])

        logits,_ =vgg.vgg_16(x,num_classes=5,is_training=True)
        loss = slim.losses.softmax_cross_entropy(logits,y_gt)
        accuracy = tools.accuracy(logits,y_gt)



        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies([tf.group(*update_ops)]):
        #     my_global_step = tf.Variable(0, name='global_step', trainable=False)
        #     train_op = optimizer.minimize(loss,my_global_step)


        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # my_global_step = tf.Variable(0, name='global_step', trainable=False)
        # with tf.control_dependencies(update_ops):
        #     train_op = optimizer.minimize(loss,my_global_step)




        # var_list = tf.trainable_variables()
        # g_list = tf.global_variables()
        # bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        # bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        # var_list += bn_moving_vars
        # saver = tf.train.Saver(var_list,max_to_keep=5)

        train_op = tools.train_model(loss)
        saver = tf.train.Saver(tf.global_variables())


        summary_op = tf.summary.merge_all()


        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        MAX_STEP = 6000


    try:
        for step in np.arange(MAX_STEP):

            if coord.should_stop():
                break
            tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_gt: tra_labels})


            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                # tf.summary.scalar('loss', tra_loss)
                # tf.summary.scalar('accuracy', tra_acc)
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(model_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        tra_summary_writer.close()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()