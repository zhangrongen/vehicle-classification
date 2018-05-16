# -*- coding: UTF-8 -*-
# Author: jiangtao
#Project: 5_classes
#File: gen_test_batch.py

import os
import glob
import numpy as np
import tensorflow as tf

def get_file(file_dir):
    temps = []
    images = []
    labels = []

    for root, sub_folders, files in os.walk(file_dir):
        for name in sub_folders:
            temps.append(os.path.join(root, name))

    for onefolder in temps:

        filenameIndir = glob.glob(onefolder + '/*.jpg')

        # n = len(filenameIndir)
        # n_plus = (1500 - n)
        # n_list = np.random.randint(n, size=n_plus)
        #
        # filenameIndir_plus = []
        #
        # for i in n_list:
        #     filenameIndir_plus.append(filenameIndir[i])
        #
        # filenameIndir += filenameIndir_plus


        for i in range(len(filenameIndir)):
            images.append(filenameIndir[i])
        n_img = len(filenameIndir)
        lab = onefolder.split('_')[0].split('/')[-1]

        if lab == '1':
            labels = np.append(labels, n_img * [0])
        if lab == '2':
            labels = np.append(labels, n_img * [1])
        if lab == '3':
            labels = np.append(labels, n_img * [2])
        if lab == '4':
            labels = np.append(labels, n_img * [3])
        if lab == '5':
            labels = np.append(labels, n_img * [4])


    temp = np.array([images, labels])
    temp = temp.transpose()
    #    print temp
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    # print len(image_list), len(label_list)

    return image_list, label_list

def get_batch(image,label):


    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,3)
    image = tf.image.resize_images(image,[224,224],method=1)
    # if np.random.random_sample() > 0.5:
    #     image = tf.image.random_contrast(image,0.25,0.4)
    # image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image,label],batch_size = 32,
                                              num_threads=32,capacity=16)

    image_batch = tf.cast(image_batch,tf.float32)

    label_batch = tf.one_hot(label_batch, depth = 5)
    label_batch = tf.cast(label_batch, tf.float32)

    return image_batch,label_batch

file_dir = '/home/jiangtao/datasets/DANdata/train/'

a,b = get_file(file_dir)

c,d = get_batch(a,b)

a = 5

