
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[6]:

#使用numpy生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]        #从-0.5到0.5，200个点,加一个维度，200行1列
noise = np.random.normal(0,0.02,x_data.shape)           #Gaussian distribution. (loc=0.0, scale=1.0, size=None)
y_data = np.square(x_data) + noise

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])             #1列
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1          #x*Weights_L1+biases_L 信号总和
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))           #中间层10个，输出1个
biases_L2 = tf.Variable(tf.zeros([1,1]))                    #输出层1个，所以有一个神经元
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)              #打印样本点
    plt.plot(x_data,prediction_value,'r-',lw=5)         #红线实现，线宽5
    plt.show()


# In[ ]:



