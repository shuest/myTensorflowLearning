import tensorflow as tf

with tf.variable_scope("scope"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v",[1])
        x = 1.0 +v
print("v:",v.name)
print("x.op",x.op.name)