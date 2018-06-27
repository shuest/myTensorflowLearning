import tensorflow as tf
queue = tf.FIFOQueue(100,"float")   #创建长度为100的队列
c=tf.Variable(0.0)
op = tf.assign_add(c,tf.constant(1.0))      #+1
enqueue_op = queue.enqueue(c)       #+1结果入队

#创建队列管理器
qr = tf.train.QueueRunner(queue,enqueue_ops=[op,enqueue_op])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  #协调器
    #启动入队线程
    enqueue_threads = qr.create_threads(sess,coord=coord,start=True)
    for i in range (0,10):
        print("-------------")
        print(sess.run(queue.dequeue()))
    coord.request_stop()    #通知其他线程关闭