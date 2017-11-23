import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from collections import Counter


#训练集数据的引入
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/srhyme/ML project/DS/train.tfrecords"])    
_, example = reader.read(filename_queue)  
features = tf.parse_single_example(
    example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),  
        'label': tf.FixedLenFeature([], tf.int64),
    })
train_images = tf.decode_raw(features['image_raw'], tf.uint8)
train_labels = tf.cast(features['label'], tf.int32)
train_pixels = tf.cast(features['pixels'], tf.int32) 


#测试集数据的引入
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/srhyme/ML project/DS/test.tfrecords"])    
_, example = reader.read(filename_queue)  
features = tf.parse_single_example(
    example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),  
        'label': tf.FixedLenFeature([], tf.int64),
    })
test_images = tf.decode_raw(features['image_raw'], tf.uint8)
test_labels = tf.cast(features['label'], tf.int32)
test_pixels = tf.cast(features['pixels'], tf.int32)


#设置变量
testnum=int(input('请输入需要测试的数据集数量:'))
k=5
correct_probability=testnum

with tf.Session() as sess:  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    test_num=sess.run(test_pixels)
    train_num=sess.run(train_pixels)


    #生成一个训练标签的列表,方便索引
    c_labels=[]
    for n in range(train_num):
        train_label=sess.run(train_labels)
        c_labels.append(train_label)
    #生成一个测试标签的列表,方便索引
    g_labels=[]
    for n in range(test_num):
        test_label=sess.run(test_labels)
        g_labels.append(test_label)


    for i in range(testnum):#测试集数量
        test_image=sess.run(test_images)
        #生成每个测试集的距离列表
        min_label=[]
        for j in range(train_num):#训练集数量
            train_image=sess.run(train_images)
            euclidean_distance =np.sqrt(np.sum(np.square(train_image - test_image)))
            min_label.append(-euclidean_distance)
            

        #生成最近k个点的位置
        min_labels=tf.constant([min_label])
        _, nearest_index = tf.nn.top_k(min_labels, k)
        #生成一个最近k点标签列表
        nearest_label=[]
        near=nearest_index
        for m in range(k):
            nearest_label.append(c_labels[sess.run(near[0,m])])#在训练标签中找到该位置的标签
        

        #生成该测试集经过knn训练后拟合的标签
        nearset_dict=Counter(nearest_label)
        
        key_list=[]
        value_list=[]
        for key,value in nearset_dict.items():
            key_list.append(key)
            value_list.append(value)
        max_value=max(value_list)
        get_value_index = value_list.index(max_value)
        guess = key_list[get_value_index]

        #判断正确率
        correct=g_labels[i]
        if correct != guess:
            correct_probability=correct_probability - 1

print('正确率为',(correct_probability/testnum))