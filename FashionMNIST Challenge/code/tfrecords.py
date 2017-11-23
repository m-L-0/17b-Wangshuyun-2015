# 将MNIST输入数据转化为TFRecord的格式
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
  
# 把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#读取fashion-mnist训练集数据
mnist = input_data.read_data_sets("/home/srhyme/fashion-mnist-master/data/fashion", dtype=tf.uint8, one_hot=True)  
#训练数据的图像,所对应的正确答案以及图像分辨率，作为一个属性来存储  
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[0]  
#训练数据的个数
num_examples = mnist.train.num_examples
#指定要写入TFRecord文件的地址
filename = "/home/srhyme/ML project/DS/train.tfrecords"  
#创建一个write来写TFRecord文件  
writer = tf.python_io.TFRecordWriter(filename)  
for index in range(num_examples):  
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构  
    example = tf.train.Example(features=tf.train.Features(feature={  
        'pixels': _int64_feature(pixels),  
        'label': _int64_feature(np.argmax(labels[index])),  
        'image_raw': _bytes_feature(image_raw)}))  
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())   
writer.close()
print('训练集已经全部写入，写入数量为：',num_examples)


#读取fashion-mnist测试集数据
mnist = input_data.read_data_sets("/home/srhyme/fashion-mnist-master/data/fashion", dtype=tf.uint8, one_hot=True)  
#测试数据的图像,所对应的正确答案以及图像分辨率，作为一个属性来存储  
images = mnist.test.images
labels = mnist.test.labels
pixels = images.shape[0]  
#测试数据的个数
num_examples = mnist.test.num_examples
#指定要写入TFRecord文件的地址
filename = "/home/srhyme/ML project/DS/test.tfrecords"  
#创建一个write来写TFRecord文件  无标题文档
writer = tf.python_io.TFRecordWriter(filename)  
for index in range(num_examples):  
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构  
    example = tf.train.Example(features=tf.train.Features(feature={  
        'pixels': _int64_feature(pixels),  
        'label': _int64_feature(np.argmax(labels[index])),  
        'image_raw': _bytes_feature(image_raw)}))  
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString()) 
writer.close()
print('测试集已经全部写入，写入数量为：',num_examples)