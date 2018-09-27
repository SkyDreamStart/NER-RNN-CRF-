# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:12:22 2018

@author: root
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib import rnn

# 加载数据

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

trainimgs, trainlabels, testimgs, testlabels \
= mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

ntrain, ntest, dim, nclasses\
=trainimgs.shape[0],testimgs.shape[0],trainimgs.shape[1],trainlabels.shape[1]

#print(ntrain, ntest, dim, nclasses)

print ("MNIST loaded")

#设置参数,权重，偏置

diminput = 28

dimhidden = 128

dimoutput = nclasses

nsteps = 28

W = {"h1" : tf.Variable(tf.random_normal([diminput,dimhidden])),

    "h2" : tf.Variable(tf.random_normal([dimhidden,dimoutput]))}

b = {"b1" : tf.Variable(tf.random_normal([dimhidden])),

    "b2" : tf.Variable(tf.random_normal([dimoutput]))}

# 创建模型

def RNN(X,W,b,nsteps):

    X = tf.transpose(X,[1,0,2])

    X = tf.reshape(X,[-1,diminput])

    H_1 = tf.matmul(X,W["h1"])+b["b1"]

    H_1 = tf.split(H_1,nsteps,0)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden,forget_bias=1.0)

    LSTM_O,LSTM_S = rnn.static_rnn(lstm_cell,H_1,dtype=tf.float32)

    O = tf.matmul(LSTM_O[-1],W["h2"])+b["b2"]

    return {"X":X,"H_1":H_1,"LSTM_O":LSTM_O,"LSTM_S":LSTM_S,"O":O} 

print ("Network ready")

# 设置损失，优化,学习率，准确率，参数初始化

learning_rate = 0.001

x      = tf.placeholder("float", [None, nsteps, diminput])

y      = tf.placeholder("float", [None, dimoutput])

myrnn  = RNN(x, W, b, nsteps)

pred  = myrnn['O']

cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))

optm  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

accr  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))

init  = tf.global_variables_initializer()

print ("Network Ready!")

# 训练，测试

#所有样本迭代（epoch）5次

training_epochs = 5

#每进行一次迭代选择的样本数

batch_size      = 16

#展示

display_step    = 1

sess = tf.Session()

sess.run(init)

print ("Start optimization")

for epoch in range(training_epochs):

    avg_cost = 0.

    total_batch = int(mnist.train.num_examples/batch_size)

    #total_batch = 100

    # Loop over all batches

    for i in range(total_batch):

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))

        # Fit training using batch data

        feeds = {x: batch_xs, y: batch_ys}

        sess.run(optm, feed_dict=feeds)

        # Compute average loss

        avg_cost += sess.run(cost, feed_dict=feeds)/total_batch

    # Display logs per epoch step

    if epoch % display_step == 0:

        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))

        feeds = {x: batch_xs, y: batch_ys}

        train_acc = sess.run(accr, feed_dict=feeds)

        print (" Training accuracy: %.3f" % (train_acc))

        testimgs = testimgs.reshape((ntest, nsteps, diminput))

        feeds = {x: testimgs, y: testlabels}

        test_acc = sess.run(accr, feed_dict=feeds)

        print (" Test accuracy: %.3f" % (test_acc))

print ("Optimization Finished.")