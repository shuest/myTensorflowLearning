import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize":[],"loss":[]}
def moving_average(a,w=10) :
    if len(a) < w :
        return a[:]
    return [val if idx< w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]
    #如果长度小于w,就是a。否则，如果a[idx]，的idx<w,就返回其值val,否则前w个保留，取【idx-w，idx]的和/w

train_X = np.linspace(-1,1,100)
train_Y = 2*train_X+np.random.randn(*train_X.shape)*0.3

plt.plot(train_X,train_Y,'ro', label = 'original data')
plt.legend()
plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")
z = tf.multiply(X,W)+b

cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
training_epochs = 20
display_step = 2
saver = tf.train.Saver(max_to_keep=1)
savedir = "log/"

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        if epoch % display_step ==0:
            loss = sess.run(cost,feed_dict={X:x,Y:y})
        print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
        if not (loss=="NA"):
            plotdata["batchsize"].append(epoch)
            plotdata["loss"].append(loss)
        saver.save(sess,savedir+"linemodel2.cpkt",global_step=epoch)
    print("Finished")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))

    #显示
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted Wline')
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Trainong loss')
    plt.show()

    load_epoch = 18
    with tf.Session() as sess2:
        sess2.run(tf.global_variables_initializer())
        saver.restore(sess2,savedir+"linermodel2.ckpt-"+str(load_epoch))
        print("x=0.2,z=",sess2.run(z,feed_dict={X:0.2}))

        kpt = tf.train.latest_checkpoint(savedir)
            if kpt!=None:
                saver.restore(sess,kpt)