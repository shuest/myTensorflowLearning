from tensorflow.examples.tutorials.mnist import input_data
import pylab
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])   #MNIST数据集维度是28*28=784
y = tf.placeholder(tf.float32,[None,10])    #数字0~9

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,W)+b)  #输出节点

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))   #pred与y交叉熵，取平均值
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25    #训练集迭代25次
batch_size = 100        #训练过程中一次取100条进行训练
display_step = 1        #每训练一次把具体的中间状态显示出来

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost +=c/total_batch
        if (epoch+1)%display_step ==0:
            print("Epoch:", '%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("finish")
