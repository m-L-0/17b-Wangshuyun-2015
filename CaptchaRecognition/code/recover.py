import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#训练集数据的引入
def apirun(img):
    #初始化权重
    def weight (shape):
        temp = tf.truncated_normal(shape=shape, stddev = 0.1)
        return tf.Variable(temp)

    #初始化偏置值
    def bias (shape):
        temp = tf.constant(0.1, shape = shape)
        return tf.Variable(temp)

    #卷积,步长为1,采用SAME边界处理
    def convolution (data,weight):
        return tf.nn.conv2d(data,weight,strides=[1,1,1,1],padding='SAME')

    #最大池化,步长为2,采用SAME边界处理,滑动窗为2*2
    def pooling (data):
        return tf.nn.max_pool(data,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #定义输入数据,其中None,-1代表数量不定,
    x=tf.placeholder(tf.float32,[None,1989])
    data_image = tf.reshape(x,[-1,39,51,1])
    #第一层:一次卷积一次池化
    w_1=weight([5,5,1,32])
    b_1=bias([32])
    #使用relu激活函数处理数据
    d_conv1=tf.nn.relu(convolution(data_image,w_1)+b_1)
    d_pool1=pooling(d_conv1)

    #第二层:一次卷积一次池化
    w_2=weight([5,5,32,64])
    b_2=bias([64])
    d_conv2=tf.nn.relu(convolution(d_pool1,w_2)+b_2)
    d_pool2=pooling(d_conv2)

    #第三层:全连接
    w_3=weight([10*13*64,1024])
    b_3=bias([1024])
    d_3=tf.reshape(d_pool2,[-1,10*13*64])
    d_fc3=tf.nn.relu(tf.matmul(d_3,w_3)+b_3)

    #第四层:softmax输出
    w_4=[weight([1024,11])for i in range(4)]
    b_4=[bias([11])for i in range(4)]
    d_4=[tf.nn.softmax(tf.matmul(d_fc3,w_4[i])+b_4[i])for i in range(4)]

    #预测标签
    finnal=tf.argmax(d_4,2)

    saver=tf.train.Saver(max_to_keep=1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '/home/srhyme/ML project/CaptchaRecognition/code/ckpt/-5244')
    predction_list =finnal.eval(session = sess,
                                        feed_dict = {x:img/255})
    predction_label=''
    for i in predction_list:
        if i[0]<10:
            predction_label = predction_label+str(i[0])
    return predction_label