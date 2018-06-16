from tensorflow.examples.tutorials.mnist import input_data
import pylab
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
print('输入数据：',mnist.train.images)
print('输入数据打shape',mnist.train.images.shape)
im = mnist.train.images[3]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder()
y = tf.placeholder()
pred = tf.nn.softmax(tf.matmul(x,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)