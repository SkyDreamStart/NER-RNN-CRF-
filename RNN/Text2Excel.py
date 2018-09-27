# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 20:49:06 2018

@author: root
"""
import xlwt
# =============================================================================
# 读取文本，将其处理为字典
f = open('eng.train.openNLP',encoding='utf-8')
f1 = open('context_train.txt','w',encoding='utf-8')
line = f.readline()
#lineDirect = {}
lineList = []
lineCount = 0
while line:
     line = line.replace('\n','')
     line = line.split(' ')
     lineList.append(line)
     #lineDirect = {lineCount : lineList}
     #print(line)
     f1.write(line[0]+' ')
     lineCount += 1
     line = f.readline() 
f.close()
f1.close()
# =============================================================================
     
# =============================================================================
#创建一个Workbook对象 编码encoding
work_book = xlwt.Workbook(encoding='utf-8',style_compression=0)
#添加一个tabel工作表
table = work_book.add_sheet('test',cell_overwrite_ok=True)
#定义一个待写入列表
headings = ['word','label']
j=0
for oneList in lineList:
    for i in range(len(oneList)):
        table.write(j,i,oneList[i])
    j+=1
work_book.save('vector2.xls')

# =============================================================================
