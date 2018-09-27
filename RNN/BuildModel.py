# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 20:26:15 2018

@author: root
"""

import DataPreTreatment
import tensorflow as tf

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
trainmodel,trainWoEmbedding = DataPreTreatment.LoadW2V(trainW2VFilePath,trainWoList)
# trainWoList,trainLaList:[51577]; trainLaArray:[51577,5]; trainWoEmbedding:[51577,128]

#加载测试数据
testWoList,testLaList,testLaArray = DataPreTreatment.ReadExcel(testExcelFilePath)
testmodel,testWoEmbedding = DataPreTreatment.LoadW2V(testW2VFilePath,testWoList)
# testWoList,testLaList:[55648]; testLaArray:[55648,5]; testWoEmbedding:[55648,128]
print("Data load success!")
# =============================================================================


# =============================================================================
#设置参数、权重、偏置
diminput = 28
dimhidden = 128
nclasses = 5   #共5种标签类型
dimoutput = nclasses
nsteps = 28
W = {"w1" : tf.Variable(tf.random_normal([diminput,dimhidden])),
     "w2" : tf.Variable(tf.random_normal([2*dimhidden,dimoutput]))}
b = {"b1" : tf.Variable(tf.random_normal([dimhidden])),
     "b2" : tf.Variable(tf.random_normal([dimoutput]))}
print("set weight&bais success")
# =============================================================================
