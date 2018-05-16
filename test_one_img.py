# !/usr/bin/python2
# Author: jiangtao
#Project: no_tfrecords_using_slim
#File: test_one_img.py 
#Ide: PyCharm
# _*_ coding: utf-8 _*_

#%% Evaluate one image
# when training, comment the following codes.

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.slim.nets import vgg,resnet_v2



def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   model_dir = '/home/jiangtao/model/5_class_resnet/'
   img_dir = '/home/jiangtao/datasets/DANdata/test/2_test/Audi0001.jpg'
   image = Image.open(img_dir)
   plt.imshow(image)
   image = image.resize([224, 224])
   image = np.array(image)
   print(image.shape)
   image_array = image.reshape([224,224,3])
   image_tensor = tf.constant(image_array)
   image_tensor_std = tf.image.per_image_standardization(image_tensor)
   with tf.Session() as sess:
        image_array = image_tensor_std.eval()
   image_array = image_array.reshape([1,224,224,3])



   with tf.Graph().as_default():


       x = tf.placeholder(tf.float32, shape=[1,224, 224, 3])
       logit,_ = resnet_v2.resnet_v2_101(x,num_classes=5,is_training=False)
       logit = tf.nn.softmax(logit)

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

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)

           print(max_index)

           # if max_index == 1:
           #     print('this is audi')
           # if max_index == 2:
           #     print('this is Benz')
           # if max_index == 3:
           #     print('this is BjHyundai')
           # if max_index == 4:
           #     print('this is BMW')
           # if max_index == 5:
           #     print('this is Buick')
           # if max_index == 6:
           #     print('this is BYD')
           # if max_index == 7:
           #     print('this is Changan')
           # if max_index == 8:
           #     print('this is Chery')
           # if max_index == 9:
           #     print('this is Chevrolet')
           # if max_index == 10:
           #     print('this is Citroen')
           # if max_index == 11:
           #     print('this is Ford')
           # if max_index == 12:
           #     print('this is Geely')
           # if max_index == 13:
           #     print('this is Honda')
           # if max_index == 14:
           #     print('this is Kia')
           # if max_index == 15:
           #     print('this is Lexus')
           # if max_index == 16:
           #     print('this is Mazda')
           # if max_index == 17:
           #     print('this is Nissan')
           # if max_index == 18:
           #     print('this is Peugeot')
           # if max_index == 19:
           #     print('this is Skoda')
           # if max_index == 20:
           #     print('this is Toyota')
           # if max_index == 21:
           #     print('this is VolksWagen')
           # if max_index == 0:
           #     print('this is Volvo')

if __name__ == '__main__':
    evaluate_one_image()