import tensorflow as tf
import pylab
with tf.Session() as sess:
    tf.global_variables_initializer()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)
    image_batch,label_batch = sess.run([images_test,labels_test])
    print("_\n",image_batch[0])
    print("_\n",label_batch[0])
    pylab.imshow(image_batch[0])
    pylab.show()
    coord.request_stop()