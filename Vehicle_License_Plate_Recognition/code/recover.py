import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

bq = {0: '云',1: '京',2: '冀',3: '吉',4: '宁',5: '川',6: '广',7: '新',8: '晋',9: '桂',10: '沪',11: '津',12: '浙',13: '渝',14: '湘',15: '琼',16: '甘',17: '皖',18: '粤',19: '苏',20: '蒙',21: '藏',22: '豫',23: '贵',24: '赣',25: '辽',26: '鄂',27: '闽',28: '陕',29: '青',30: '鲁',31: '黑'}

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/srhyme/车牌字符识别训练数据/DS/test.tfrecords"])    
_, example = reader.read(filename_queue) 
features = tf.parse_single_example(
    example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),  
        'label': tf.FixedLenFeature([], tf.int64),
    })
test_images = tf.decode_raw(features['image_raw'], tf.uint8)
test_labels = tf.cast(features['label'], tf.int32)

#数据集的预处理
with tf.Session() as sess:
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    test_image=[]
    test_label=[]
    for i in range(1114):
        image,label=sess.run([test_images,test_labels])
        image=image.reshape((1,784))
        temp=np.zeros((1,32))
        temp[0][label]=1
        temp=temp.reshape((1,32))
        test_image.append(image/255)
        test_label.append(temp)
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
w_4=weight([1024,32])
b_4=bias([32])
d_4=tf.nn.softmax(tf.matmul(d_fc3_drop,w_4)+b_4)

#定义损失函数(交叉熵),并用ADAM优化器优化
y = tf.placeholder("float", [None,32])
loss_function = - tf.reduce_sum(y * tf.log(d_4))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss_function)

#判断预测标签和实际标签是否匹配
correct = tf.equal(tf.argmax(d_4,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,"float"))


saver=tf.train.Saver(max_to_keep=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()  
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
ckpt = tf.train.get_checkpoint_state('./ckpt')
saver.restore(sess, ckpt.model_checkpoint_path)
# test_accuracy = accuracy.eval(session = sess,
#                                         feed_dict = {x:test_image, y:test_label, keep_prob:1.0})
# print("test_accuracy %g" % test_accuracy)

recall_dict={i:[0,0] for i in range(32)}
for i in range(1114):
    forecast_label=tf.argmax(d_4,1)
    true_label=tf.argmax(y,1)
    a=forecast_label.eval(session = sess,
                                    feed_dict={x: test_image[i], y: test_label[i], keep_prob: 1.0})
    b=true_label.eval(session = sess,
                                    feed_dict={x: test_image[i], y: test_label[i], keep_prob: 1.0})
    if a==b:
        recall_dict[b[0]][0]+=1
        recall_dict[b[0]][1]+=1
    if a!=b:
        recall_dict[b[0]][0]+=1
all=0
for i in range(32):
    if recall_dict[i][1]==0:
        print(bq[i],'的召回率为:0')
    else:
        print(bq[i],'的召回率为:',recall_dict[i][1]/recall_dict[i][0])
    all+=recall_dict[i][1]
print('准确率为:',all/1114)