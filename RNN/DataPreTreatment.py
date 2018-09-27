# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:59:17 2018

@author: root
"""

import xlrd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# =============================================================================
# 读取excel
def ReadExcel(path):
    ExcelFile = xlrd.open_workbook(path)
    #print(ExcelFile.sheet_names()[2])   根据索引打印表名
    sheet=ExcelFile.sheet_by_name('TestData')
    #print(sheet.name,sheet.nrows)    #根据索引打印表内容
    colOneList = sheet.col_values(0)         #读取表格第一列内容，长度40716
    colTwoList = sheet.col_values(1)         #读取表格第二列内容，长度40716  
    labelArray = np.zeros((len(colOneList),5))
    for i in range(len(colOneList)):
        if colTwoList[i] == "O":
            labelArray[i][0] = 1
        elif colTwoList[i] == "I-ORG":
            labelArray[i][1] = 1
        elif colTwoList[i] == "I-LOC":
            labelArray[i][2] = 1
        elif colTwoList[i] == "I-MISC":
            labelArray[i][3] = 1
        else:
            labelArray[i][4] = 1
    return colOneList,colTwoList,labelArray
# =============================================================================
    

# =============================================================================
# 构建词向量
# 模型训练
def CreatW2V(path,savepath):
    sentence = LineSentence(path)
    #128维，窗口大小为5
    model = Word2Vec(sentence,size=128,window=5,min_count=1,workers=4)
    model.save(savepath)
    #Word2Vec(vocab=9967, size=128, alpha=0.025)  (9967,128)
# =============================================================================
#加载词向量
def LoadW2V(path,wordList):
    model = Word2Vec.load(path)
    wordEmbedding = []
    for i in range(len(wordList)):
        wordEmbedding.append(model[wordList[i]])
    wordEmbedding = np.array(wordEmbedding)
    return model,wordEmbedding
# =============================================================================


# =============================================================================
if __name__ == '__main__':
    excelFilePath = 'vector2.xls'
    textFliePath = 'context_train.txt'
    W2VFilePath = 'train_word_embedding_128'
    colOneList,colTwoList,labelArray = ReadExcel(excelFilePath)
    CreatW2V(textFliePath,W2VFilePath)
#    model,wordEmbedding = LoadW2V(W2VFilePath,colOneList)
#    print(colOneList,labelArray)
#    model = Word2Vec.load(W2VFilePath)
#    print(model)
#    print(model['Zurich'])