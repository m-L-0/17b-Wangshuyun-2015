from PIL import Image
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
import re

# 把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
  
# 把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



#获取图片名字列表
nums = os.listdir('/home/srhyme/ML project/CaptchaRecognition/captcha/images')
#利用正则化截取字符串中的数字用来遍历其中的label
label_list=[]
f=open("/home/srhyme/ML project/CaptchaRecognition/captcha/labels/labels.csv")
temp=list(f.readlines())
for i in temp:
    label_list.append(re.findall(r"\d+\d?\d*", i)[1])


statistics_list=[]
one_len=0
two_len=0
three_len=0
four_len=0
#统计数据
print("数据集总数为:",len(nums))
for i in range(len(nums)):
    statistics_list.append(label_list[int(re.findall(r"\d+\d?\d*", nums[i])[0])])
for i in statistics_list:
    if len(i)==1:
        one_len+=1
    if len(i)==2:
        two_len+=1
    if len(i)==3:
        three_len+=1
    if len(i)==4:
        four_len+=1

print("一位数据的数量为:",one_len,",所占比例:",one_len/400,"%")
print("两位数据的数量为:",two_len,",所占比例:",two_len/400,"%")
print("三位数据的数量为:",three_len,",所占比例:",three_len/400,"%")
print("四位数据的数量为:",four_len,",所占比例:",four_len/400,"%")

#训练集生成
images=[]
labels=[]
for i in range(32000):
    im=Image.open('/home/srhyme/ML project/CaptchaRecognition/captcha/images/'+nums[i])
    im = im.resize((51,39))
    im = im.convert('L')
    im=np.array(im)
    images.append(im)
    temp=re.findall(r"\d+\d?\d*", nums[i])[0]
    labels.append(int(label_list[int(temp)]))
images=np.array(images)
labels=np.array(labels)

writer = tf.python_io.TFRecordWriter('/home/srhyme/ML project/CaptchaRecognition/DS/train.tfrecords')
for index in range(images.shape[0]): 
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构  
    example = tf.train.Example(features=tf.train.Features(feature={  
        'label': _int64_feature(labels[index]),  
        'image_raw': _bytes_feature(image_raw)}))  
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())   
writer.close()
print('训练集已写入完毕,写入数量为',images.shape[0])




#验证集生成
images=[]
labels=[]
for i in range(32000,36000):
    im=Image.open('/home/srhyme/ML project/CaptchaRecognition/captcha/images/'+nums[i])
    im = im.resize((51,39))
    im = im.convert('L')
    im=np.array(im)
    images.append(im)
    temp=re.findall(r"\d+\d?\d*", nums[i])[0]
    labels.append(int(label_list[int(temp)]))
images=np.array(images)
labels=np.array(labels)

writer = tf.python_io.TFRecordWriter('/home/srhyme/ML project/CaptchaRecognition/DS/validation.tfrecords')
for index in range(images.shape[0]): 
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构  
    example = tf.train.Example(features=tf.train.Features(feature={  
        'label': _int64_feature(labels[index]),  
        'image_raw': _bytes_feature(image_raw)}))  
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())   
writer.close()
print('验证集已写入完毕,写入数量为',images.shape[0])




#测试集生成
images=[]
labels=[]
for i in range(36000,40000):
    im=Image.open('/home/srhyme/ML project/CaptchaRecognition/captcha/images/'+nums[i])
    im = im.resize((51,39))
    im = im.convert('L')
    im=np.array(im)
    images.append(im)
    temp=re.findall(r"\d+\d?\d*", nums[i])[0]
    labels.append(int(label_list[int(temp)]))
images=np.array(images)
labels=np.array(labels)

writer = tf.python_io.TFRecordWriter('/home/srhyme/ML project/CaptchaRecognition/DS/test.tfrecords')
for index in range(images.shape[0]): 
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构  
    example = tf.train.Example(features=tf.train.Features(feature={  
        'label': _int64_feature(labels[index]),  
        'image_raw': _bytes_feature(image_raw)}))  
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())   
writer.close()
print('测试集已写入完毕,写入数量为',images.shape[0])
