# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 20:14:37 2018

@author: zhanglisama    jxufe
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from time import time
#ªÒ»°dataset
def load_data(dataset_path):
    
    X = []
    
    for dirname,dirnames,filensmes in os.walk(dataset_path):
        for subdirname in dirnames:   
            subject_path=os.path.join(dirname,subdirname)  
            for filename in os.listdir(subject_path):   
                im=Image.open(os.path.join(subject_path,filename))
                im=np.asarray(im,dtype=np.float32)/256
                im=np.ndarray.flatten(im)
                X.append(im)
                
    label = np.zeros((400, 40))
    for i in range(40):
        label[i * 10: (i + 1) * 10, i] = 1
        
    images_train,images_test,target_train,target_test=train_test_split(X,label,test_size=0.2,random_state=0)
    return [(images_train,images_test),(target_train,target_test)]
t0 = time()
dataset_path = 'att_faces'
dataset = load_data(dataset_path)
batch_size = 40
train_set_x = dataset[0][0]   
train_set_x =np.asarray(train_set_x, dtype='float32')
#train_set_x = np.reshape(train_set_x,[-1,112,92])
train_set_y = dataset[1][0]    #—µ¡∑±Í«©
valid_set_x = dataset[0][1]
valid_set_x = np.asarray(valid_set_x, dtype='float32')
#valid_set_x = np.reshape(valid_set_x,[-1,112,92])
valid_set_y = dataset[1][1]

X = tf.placeholder(tf.float32,[None,10304])
Y = tf.placeholder(tf.float32,[None,40])

w1 = tf.Variable(tf.truncated_normal([10304,200],stddev=0.1))
b1 = tf.Variable(tf.zeros([200]))
output1 = tf.nn.swish(tf.matmul(X,w1)+b1)

w2 = tf.Variable(tf.truncated_normal([200,100],stddev=0.1))
b2 = tf.Variable(tf.zeros([100]))
output2 = tf.nn.swish((tf.matmul(output1,w2)+b2))

w3 = tf.Variable(tf.truncated_normal([100,40],stddev=0.1))
b3 = tf.Variable(tf.zeros([40]))
output = tf.matmul(output2,w3)+b3
predict = tf.nn.softmax(output)

loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(predict),reduction_indices=[1]))
optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)

correct = tf.equal(tf.arg_max(Y,1),tf.arg_max(predict,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

keep_prob=tf.placeholder(tf.float32)
max_iter = 50

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_iter):
        epoch_loss = 0
        for j in range(int(train_set_x.shape[0]/batch_size)):
            x = train_set_x[j*batch_size:(j+1)*batch_size]
            y = train_set_y[j*batch_size:(j+1)*batch_size]
            
            _,cost = sess.run([optimizer,loss],feed_dict = {X:x,Y:y,keep_prob:0.75})
            epoch_loss+=cost
        print('train accuracy: ',accuracy.eval({X:x,Y:y}),' ','step: ',i,' ','loss:',cost)
        
    print('valid accuracy:',accuracy.eval({X:valid_set_x,Y:valid_set_y,keep_prob:1}))
    print('time:%f'%(time()-t0))
    print('finish')