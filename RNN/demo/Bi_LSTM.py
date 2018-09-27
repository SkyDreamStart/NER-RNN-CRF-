"""
Created on Fri Sep 21 09:14:51 2018

@author: root
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# =============================================================================
#加载数据
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
trainimgs, trainlabels, testimgs, testlabels \
= mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

ntrain, ntest, dim, nclasses\
=trainimgs.shape[0],testimgs.shape[0],trainimgs.shape[1],trainlabels.shape[1]

print(trainimgs.shape,trainlabels.shape,testimgs.shape,testlabels.shape)
#(55000, 784) (55000, 10) (10000, 784) (10000, 10)
print(ntrain, ntest, dim, nclasses)
#55000 10000 784 10
print ("MNIST loaded") 
# =============================================================================

# =============================================================================
#设置参数、权重、偏置
diminput = 28
dimhidden = 128
dimoutput = nclasses
nsteps = 28
W = {"w1" : tf.Variable(tf.random_normal([diminput,dimhidden])),
     "w2" : tf.Variable(tf.random_normal([2*dimhidden,dimoutput]))}
b = {"b1" : tf.Variable(tf.random_normal([dimhidden])),
     "b2" : tf.Variable(tf.random_normal([dimoutput]))}
print("set weight&bais success")
# =============================================================================

# =============================================================================
# 模型建立
def Bi_RNN(X,W,b,nsteps,name):
    X = tf.transpose(X,[1,0,2])
    X = tf.reshape(X,[-1,diminput])
    H_1 = tf.matmul(X,W["w1"])+b["b1"]
    H_1 = tf.split(H_1,nsteps,0) 
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden,forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden,forget_bias=1.0)
    outputs,_,_ = rnn.static_bidirectional_rnn(
            lstm_fw_cell,
            lstm_bw_cell,
            H_1,
            dtype=tf.float32)
    return tf.matmul(outputs[-1],W["w2"])+b["b2"]
print ("Network ready")
# =============================================================================

# =============================================================================
# 设置损失，优化,学习率，准确率，参数初始化
learning_rate = 0.001
x = tf.placeholder("float", [None, nsteps, diminput])
y = tf.placeholder("float", [None, dimoutput])
myrnn  = Bi_RNN(x, W, b, nsteps,'basic')
pred  = myrnn
cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optm  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
accr  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))
init  = tf.global_variables_initializer()
print ("Network Ready!")
# =============================================================================


# =============================================================================
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
# ============================================================================

