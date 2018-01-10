import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#训练集数据的引入
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/srhyme/ML project/CaptchaRecognition/DS/train.tfrecords"])    
_, example = reader.read(filename_queue) 
features = tf.parse_single_example(
    example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),  
        'label': tf.FixedLenFeature([], tf.int64),
    })
train_images = tf.decode_raw(features['image_raw'], tf.uint8)
train_labels = tf.cast(features['label'], tf.int32)

#数据集的预处理
with tf.Session() as sess:
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_image=[]
    train_label=[]
    temp_image=[]
    temp_label=[]
    for i in range(32000):
        image,label=sess.run([train_images,train_labels])
        temp_image.append(image)
        temp=np.zeros((4,11))
        for k in range(len(str(label))):    
            temp[k][int(str(label)[k])] = 1
        for l in range(len(str(label)),4):
            temp[l][10] = 1
        temp_label.append(temp)
    for j in range(640):
        train_image.append(np.array(temp_image[j*50:j*50+50])/255)
        train_label.append(np.array(temp_label[j*50:j*50+50]))
    print('训练集已加载完毕')

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

#dropout操作,防止过拟合
keep_prob=tf.placeholder(tf.float32)
d_fc3_drop=tf.nn.dropout(d_fc3,keep_prob)

#第四层:softmax输出
w_4=[weight([1024,11])for i in range(4)]
b_4=[bias([11])for i in range(4)]
d_4=[tf.nn.softmax(tf.matmul(d_fc3_drop,w_4[i])+b_4[i])for i in range(4)]

#定义损失函数(交叉熵),并用ADAM优化器优化
y = tf.placeholder("float", [None,4, 11])
y1 = [tf.slice(y, [0,n,0], [50,1,11]) for n in range(4)]# 2指的是batch，n指的是
y2 = tf.reshape(y1, [4,50,11])
loss_function = [- tf.reduce_sum(y2[i] * tf.log(d_4[i]+1e-8))for i in range(4)]
cross_entropy = tf.reduce_mean(loss_function)
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

#判断预测标签和实际标签是否匹配
correct = tf.equal(tf.argmax(d_4,2), tf.argmax(y2,2))
accuracy = tf.reduce_mean(tf.cast(correct,"float"))
sess = tf.Session()     
sess.run(tf.global_variables_initializer())

#实现可视化
tf.summary.scalar('loss',cross_entropy)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/home/srhyme/ML project/CaptchaRecognition/code/graph/', sess.graph)

#运行与打印
saver=tf.train.Saver(max_to_keep=1)
max_acc=0
for j in range (10):
    print('开始训练第 %d 次'%(j+1))
    for i in range(640):
        summary_str, _ = sess.run([summary_op, optimizer], feed_dict = {x:train_image[i], y:train_label[i],keep_prob:0.8})
        summary_writer.add_summary(summary_str,i)
        train_accuracy = accuracy.eval(session = sess,
                                            feed_dict = {x:train_image[i], y:train_label[i],keep_prob:0.8})
        print("step %d, train_accuracy %g" %(i+j*640, train_accuracy))

        if train_accuracy>max_acc:
            max_acc=train_accuracy
            saver.save(sess,'/home/srhyme/ML project/CaptchaRecognition/code/ckpt/',global_step=i+j*640)
sess.close()