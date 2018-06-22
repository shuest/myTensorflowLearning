import tensorflow as tf
global_step = tf.Variable(0,trainable=False)
initial_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step=global_step,decay_steps=10,decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(20):
        g,rate = sess.run([add_global,learning_rate])
        print(g,rate)


tf.constant_initializer(value)  #初始化一切提供的值
tf.random_uniform_initializer(a,b)  #从a到b均匀初始化
tf.random_normal_initializer(mean,stddev)  #用平均值和标准差初始化均匀分布
tf.random_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype=tf.float32)    #正太随机数
tf.truncated_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype=tf.float32) #截断正太分布随机数，不过只保留[mean-2*stddev,mean+2*stddev]内的随机数
tf.orthogonal_initializer(gain=1.0,dtype=tf.float32,seed=None)  #生成正交矩阵随机数

