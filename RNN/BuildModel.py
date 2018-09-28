# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 20:26:15 2018

@author: root
"""

import DataPreTreatment
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# =============================================================================
#训练数据路径
trainExcelFilePath = 'openNLP/vector1.xls'
trainTextFliePath = 'openNLP/context.txt'
trainW2VFilePath = 'Word2Vector/word_embedding_128'
#测试数据路径
testExcelFilePath = 'openNLP/vector2.xls'
testTextFliePath = 'openNLP/context_train.txt'
testW2VFilePath = 'Word2Vector/train_word_embedding_128'
#加载训练数据
trainWoList,trainLaList,trainLaArray = DataPreTreatment.ReadExcel(trainExcelFilePath)
trainWoList = np.array(trainWoList)
trainLaList = np.array(trainLaList)
trainLaArray = np.array(trainLaArray)
trainmodel,trainWoEmbedding = DataPreTreatment.LoadW2V(trainW2VFilePath,trainWoList)
# trainWoList,trainLaList:[51577]; trainLaArray:[51577,5]; trainWoEmbedding:[51577,128]

#加载测试数据
testWoList,testLaList,testLaArray = DataPreTreatment.ReadExcel(testExcelFilePath)
testWoList = np.array(testWoList)
testLaList = np.array(testLaList)
testLaArray = np.array(testLaArray)
testmodel,testWoEmbedding = DataPreTreatment.LoadW2V(testW2VFilePath,testWoList)
# testWoList,testLaList:[55648]; testLaArray:[55648,5]; testWoEmbedding:[55648,128]
print("Data load success!")
# =============================================================================


# =============================================================================
#设置参数、权重、偏置
diminput = 16
dimhidden = 128
nclasses = 5   #共5种标签类型
dimoutput = nclasses
nsteps = 8
W = {"w1" : tf.Variable(tf.random_normal([diminput,dimhidden])),
     "w2" : tf.Variable(tf.random_normal([2*dimhidden,dimoutput]))}
b = {"b1" : tf.Variable(tf.random_normal([dimhidden])),
     "b2" : tf.Variable(tf.random_normal([dimoutput]))}
print("set weight&bais success")
# =============================================================================


# =============================================================================
# 模型建立
def Bi_RNN(X,W,B,nsteps,name):
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
print("Network Created")
# =============================================================================


# =============================================================================
#设置损失，优化,学习率，准确率，参数初始化
learning_rate = 0.001
x = tf.placeholder("float",[None,nsteps,diminput])
y = tf.placeholder("float",[None,dimoutput])
myModel = Bi_RNN(x,W,b,nsteps,'basic')
pred = myModel
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)), tf.float32))
init = tf.global_variables_initializer()
print("Network Ready")
# =============================================================================


# =============================================================================
#训练、测试
##所有样本迭代次数（epoch）5次
training_epochs = 5
##没进行一次迭代选择的样本数10个
batch_size = 10
##展示步长
display_step = 1
sess = tf.Session()
sess.run(init)
print("Start Optimization")
for epoch in range(training_epochs):
      avg_cost = 0.  #初始化损失值
      total_batch = int(len(trainWoList)/batch_size)
      for i in range(total_batch):
            batch_xs = trainWoEmbedding[i*batch_size:(i+1)*batch_size,:]
            batch_ys = trainLaArray[i*batch_size:(i+1)*batch_size,:]
            batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
            # Fit training using batch data
            feeds = {x: batch_xs, y: batch_ys}
            sess.run(optm, feed_dict=feeds)
            #计算平均损失 
            avg_cost += sess.run(cost, feed_dict=feeds)/total_batch
      #展示结果
      if epoch % display_step == 0:
             print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
             feeds = {x: batch_xs, y: batch_ys}
             train_acc = sess.run(accr, feed_dict=feeds)
             print (" Training accuracy: %.3f" % (train_acc))
             testWoEmbedding = testWoEmbedding.reshape((len(testWoList), nsteps, diminput))
             feeds = {x: testWoEmbedding, y: testLaArray}
             test_acc = sess.run(accr, feed_dict=feeds)
             print (" Test accuracy: %.3f" % (test_acc))
print ("Optimization Finished.")
# =============================================================================
