import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

m1 = tf.constant([[3,3]])     #1行2列矩阵[3 3]
m2 = tf.constant([[2],[3]])   #2行1列矩阵[2 3]^T
product = tf.matmul(m1,m2)  #创建一个乘法op，矩阵相乘
print(product)              #输出         其实并没有执行

#Tensor("MatMul:0", shape=(1, 1), dtype=int32) shape是1行1列（1个数），结果是0

sess = tf.Session()     #定义会话，启用默认图
result = sess.run(product)   #run执行3个op
print(result)
sess.close()
#[[15]]

#简单方法：省了关闭session
with tf.Session() as sess:
    result=sess.run(product)
    print(result)