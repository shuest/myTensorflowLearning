import tensorflow as tf

x = tf.Variable([1,2])
y = tf.constant([3,3])
sub = tf.subtract(x,y)
add = tf.add(x,sub)

init = tf.global_variables_initializer()    #全局变量全部初始化
with tf.Session() as sess :
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
    #[-2 -1]    [-1  1]

state = tf.Variable(0,name='counter')       #创建变量，初始0
new_value = tf.add(state,1)
update = tf.assign(state,new_value)     #赋值，后给前
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5) :  #5次
        sess.run(update)
        print(sess.run(state))
