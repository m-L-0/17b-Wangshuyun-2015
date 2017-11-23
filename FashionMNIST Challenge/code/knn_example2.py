import tensorflow as tf
import numpy as np

k=7
test_num=int(input('请输入需要测试的数据数量：'))

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

#加载TFRecord测试集的数据
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

tri_list=[]
tei_list=[]
trl_list=[]
tel_list=[]

#转换TFRecord里面的类型格式
with tf.Session() as sess:  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(sess.run(train_pixels)):
        image,label=sess.run([train_images,train_labels])
        tri_list.append(image)
        trl=np.zeros((1,10))
        trl[0][label]=1
        trl_list.append(trl[0])
    train_labels=np.array(trl_list)
    train_images=np.array(tri＿list)
    print('训练集已加载完毕')

    for i in range(test_num):
        image,label=sess.run([test_images,test_labels])
        tei_list.append(image)
        tel=np.zeros((1,10))
        tel[0][label]=1
        tel_list.append(tel[0])
    test_labels=np.array(tel_list)
    test_images=np.array(tei＿list)
    print('测试集已加载完毕')
    

x_train = tf.placeholder(tf.float32)
x_test = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)

# 欧式距离
euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(x_train - x_test), 1))
# 计算最相近的k个样本的索引
_, nearest_index = tf.nn.top_k(-euclidean_distance, k)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    predicted_num = 0
    # 对每个图片进行预测
    for i in range(test_images.shape[0]):
        # 最近k个样本的标记索引
        nearest_index_res = sess.run(
            nearest_index, 
            feed_dict={
                x_train: train_images,
                y_train: train_labels,
                x_test: test_images[i]})
        # 最近k个样本的标记
        nearest_label = []
        for j in range(k):
            nearest_label.append(list(train_labels[nearest_index_res[j]]))

        predicted_class = sess.run(tf.argmax(tf.reduce_sum(nearest_label, 0), 0))
        true_class = sess.run(tf.argmax(test_labels[i]))
        if predicted_class == true_class:
            predicted_num += 1
        
        if i % 100 == 0:
            print('step is %d accuracy is %.4f' % (i, predicted_num / (i+1)))
    print('accuracy is %.4f' % (predicted_num / test_num))