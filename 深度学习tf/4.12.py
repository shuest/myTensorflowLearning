import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1,1,100)
train_Y = 2*train_X+np.random.randn(*train_X.shape)*0.3
plt.plot(train_X,train_Y,'ro',label = 'Original data')
plt.legend()
plt.show()
X = tf.placeholder(int)
Y = tf.placeholder(int)
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")
z = tf.multiply(X,W)+b
cost = tf.reduce_mean(tf.square(Y-z))
tf.reset_default_graph()    #重置图
display_step = 2
saver = tf.train.Saver()
savedir = "log/"
init = tf.global_variables_initializer()
with tf.Session as sess:
    sess.run(init)
    print("Finish")
    saver.save(sess,savedir+"linermodel.cpkt")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W))