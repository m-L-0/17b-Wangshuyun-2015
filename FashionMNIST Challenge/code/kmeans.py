import numpy as np
import tensorflow as tf
import random
import datetime

#加载TFRecord训练集的数据
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

#生成用于聚类的数据
data=[]

with tf.Session() as sess:  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    data_num = sess.run(train_pixels)
    for i in range(data_num):
        train_image=sess.run(train_images)
        data.append(train_image)
sess.close()
data=np.array(data)

#距离公式
def Distance (x,y):
    return np.linalg.norm(x-y)/10000000

#随机生成10个质心的索引
centroids_index=[]
while len(centroids_index) != 10:
    random_centroid=random.randint(0,data_num-1)
    if random_centroid not in centroids_index:
        centroids_index.append(random_centroid)

#根据索引生成对应的质心矩阵
centroids = np.empty((10,784))
for i in range(10):
    centroids[i]=data[centroids_index[i]]

#生成55000*2的矩阵，第一列存储样本点所属的族的索引值，第二列存储该点与所属族的质心的距离
cluster = np.empty((data_num,2))
cluster_times=0
begin = datetime.datetime.now()
while True:
    change_num=0
    #遍历每一个数据并初始化每个数据的距离和类别
    for i in range(data_num):
        min_distance = np.float64('inf')
        min_index=-1
        for j in range(10):
            distance=Distance(centroids[j],data[i])
            if distance < min_distance:
                min_distance = distance
                min_index = j
        #修正数据的类别和距离
        if cluster[i,0] != min_index:
            cluster[i] = min_index,min_distance
            change_num+=1
    #设定容错率为0,直到全部聚类完成才会跳出while循环
    if change_num == 0:
        break
    #更改质心
    for i in range(10):
        allidata = data[np.nonzero(cluster[:,0]==i)[0]]
        centroids[i] = np.mean(allidata, axis=0)
    cluster_times+=1
    print('已完成%d次聚类,此次共更改%d个数据' % (cluster_times,change_num))
end = datetime.datetime.now()
print('完成聚类任务,共聚类%d次,耗时'%(cluster_times),end-begin)