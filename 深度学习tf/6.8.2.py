import tensorflow as tf
x = tf.placeholder("float")
y = tf.placeholder("float")
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")

z = tf.natmul(x,W)+b
maxout = tf.reduce_max(z,axis=1,keep_dims=True)

W2 = tf.Variable(tf.truncated_normal([1,10],stddev=0.1))
b2 = tf.Variable(tf.zeros([1]))

pred = tf.nn.softmax(tf.matmul(maxout,W2)+b2)

cost = tf.reduce_mean(tf.square(y-z))
learning_rate = 0.04
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)