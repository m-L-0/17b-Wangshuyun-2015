import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

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


#数据集的预处理
with tf.Session() as sess:
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_image=[]
    train_label=[]
    temp_image=[]
    temp_label=[]
    for i in range(55000):
        image,label=sess.run([train_images,train_labels])
        temp_image.append(image)
        temp=np.zeros((1,10))
        temp[0][label]=1
        temp_label.append(temp[0])
    for j in range(1100):
        train_image.append(np.array(temp_image[j*50:j*50+50])/255)
        train_label.append(np.array(temp_label[j*50:j*50+50])/255)
    print('训练集已加载完毕')
    temp_image=[]
    temp_label=[]
    test_image=[]
    test_label=[]
    for i in range(500):
        image,label=sess.run([test_images,test_labels])
        temp_image.append(image/255)
        temp=np.zeros((1,10))
        temp[0][label]=1
        temp_label.append(temp[0])
    test_image=np.array(temp_image)
    test_label=np.array(temp_label)
    print('测试集已加载完毕')


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
x=tf.placeholder(tf.float32,[None,784])
data_image = tf.reshape(x,[-1,28,28,1])

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
w_3=weight([7*7*64,1024])
b_3=bias([1024])
d_3=tf.reshape(d_pool2,[-1,7*7*64])
d_fc3=tf.nn.relu(tf.matmul(d_3,w_3)+b_3)
#dropout操作,防止过拟合
keep_prob=tf.placeholder(tf.float32)
d_fc3_drop=tf.nn.dropout(d_fc3,keep_prob)

#第四层:softmax输出
w_4=weight([1024,10])
b_4=bias([10])
d_4=tf.nn.softmax(tf.matmul(d_fc3_drop,w_4)+b_4)

#定义损失函数(交叉熵),并用ADAM优化器优化
y = tf.placeholder("float", [None, 10])
loss_function = - tf.reduce_sum(y * tf.log(d_4))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss_function)

#判断预测标签和实际标签是否匹配
correct = tf.equal(tf.argmax(d_4,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,"float"))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#运行与打印
for i in range(1100):
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session = sess,
                                    feed_dict = {x:train_image[i], y:train_label[i], keep_prob:1.0})
        print("step %d, train_accuracy %g" %(i, train_accuracy))
    optimizer.run(session = sess, feed_dict = {x:train_image[i], y:train_label[i],
                keep_prob:0.5}) 
print("test accuracy %g" % accuracy.eval(session = sess,
    feed_dict = {x:test_image, y:test_label,
                keep_prob:1.0})) 