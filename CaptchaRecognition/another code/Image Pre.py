import numpy as np
import matplotlib.pyplot as plt
import pylab
import tensorflow as tf
from PIL import Image

#创建一个reader来读取TFRecord文件中的样例并创建输入队列
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/srhyme/ML project/CaptchaRecognition/DS/train.tfrecords"])  
#从文件中读取并解析一个样例  
_, example = reader.read(filename_queue)  
features = tf.parse_single_example(
    example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),  
        'label': tf.FixedLenFeature([], tf.int64),  
    })
#将字符串解析成图像对应的像素数组,其他数据转换成需要的数据类型
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32) 



with tf.Session() as sess:  
#启动多线程处理输入数据
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#可视化数据
    for i in range(2):
        image, label = sess.run([images, labels])
        image = image.reshape(39,51)
        plt.imshow(image)
        pylab.show()
        image = Image.fromarray(np.uint8(image*255))
        image.show()