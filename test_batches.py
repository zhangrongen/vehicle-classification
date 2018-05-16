# !/usr/bin/python2
# Author: jiangtao
#Project: no_tfrecords_using_slim
#File: test_batch.py 
#Ide: PyCharm
# _*_ coding: utf-8 _*_
import tensorflow as tf
import tools
import math
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception,vgg,resnet_v2
from gen_test_batch import get_file,get_batch
from tensorflow.python.ops import control_flow_ops

test_file_dir = '/home/jiangtao/datasets/DANdata/test/'
model_dir = '/home/jiangtao/model/5_class_vgg/'

img_batch_list, lab_batch_list = get_file(test_file_dir)

def test_batch():

    with tf.Graph().as_default():

        tra_image_batch, tra_label_batch = get_batch(img_batch_list, lab_batch_list)

        logits, _ = vgg.vgg_16(tra_image_batch, num_classes=5, is_training=False)
        # logits, _ = vgg.vgg_16(tra_image_batch, num_classes=5, is_training=False)

        loss = slim.losses.softmax_cross_entropy(logits, tra_label_batch)
        accuracy = tools.accuracy(logits, tra_label_batch)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('\nEvaluating......')

                num_step = int(math.floor(408 / 32))
                num_sample = num_step * 32
                step = 0
                total_correct = 0
                total_loss = 0
                while step < num_step and not coord.should_stop():
                    batch_accuracy = sess.run(accuracy)
                    batch_loss = sess.run(loss)
                    total_correct += np.sum(batch_accuracy)
                    total_loss += np.sum(batch_loss)
                    step += 1
                    print(batch_accuracy)
                    print(batch_loss)
                print('Total testing samples: %d' % num_sample)
                print('Average accuracy: %.2f%%' % ( total_correct / step))
                print('Average loss: %2.f' % (total_loss / step))

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


if __name__ =='__main__':
    test_batch()

