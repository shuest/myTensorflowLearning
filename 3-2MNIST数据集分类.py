import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少批次
n_batch = mnist.train.num_examples //batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])   #784列
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))     #权值
b = tf.Variable(tf.zeros[10])       #偏置值
prediction = tf.nn.softmax(tf.matmul(x,W)+b)    #预测值

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#初始化变量
init = tf.global_variables_initializer()
#结果准确率
#结果存放在bool列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))   #y真实值
#argmax返回一维张量中最大值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))   #将bool转成float32

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):     #所有图片训练21次
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})    #喂测试集图片和标签
        print("Iter"+str(epoch)+",Testing Accuracy"+str(acc))