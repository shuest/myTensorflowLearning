import tensorflow as tf

#Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as sess :
    result=sess.run([mul,add])      #fetch 同时运行多个op
    print(result)

#feed
input1 = tf.placeholder(tf.float32)     #32位占位符
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess :
    #feed的数据以字典的形式传入
    print(sess.run(output , feed_dict={input1 : [7.] , input2 : [2.]}))
    