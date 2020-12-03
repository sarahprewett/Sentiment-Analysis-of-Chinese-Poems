# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 20:23:05 2019

@author: Administrator
"""
from sklearn.model_selection import train_test_split
import smart_open
import numpy as np
import operator
stopWord=['，','。','《','》','：','；','“','”',"‘","’",'！','？']
def readFile(addr):
    fileobj=smart_open.open(addr)
    try:
        result=fileobj.read()
    finally:
        fileobj.close()
    result_list=result.split( )
    return result_list

neg_list=readFile(r'D:/02英才计划/课题后期/SentimentAnalysis/negative.txt')
pos_list=readFile(r'D:/02英才计划/课题后期/SentimentAnalysis/positive.txt')

neg_training, neg_testing = train_test_split(neg_list, test_size=.33)
pos_training, pos_testing = train_test_split(pos_list, test_size=.33)

def charNum(poemList):   #计算所有字在所有【特定情感】诗词中出现的次数
    char_num=0
    for everyPoem in poemList:
        for Char in everyPoem:
            if Char not in stopWord:
                char_num+=1
    return char_num

pos_num=charNum(pos_training)
neg_num=charNum(neg_training)
num=[pos_num,neg_num]

def getCount(Char,corpus):
    count=0
    for anyPoem in corpus:
        for anyChar in anyPoem:
            if anyChar==Char:
                count+=1
    return count

neg_count_set={}
for anyPoem in neg_training:
    for anyChar in anyPoem:
        if anyChar not in stopWord:
            neg_count_set[anyChar]=getCount(anyChar,neg_training)
            
pos_count_set={}
for anyPoem in pos_training:
    for anyChar in anyPoem:
        if anyChar not in stopWord:
            pos_count_set[anyChar]=getCount(anyChar,pos_training)
swd = sorted(pos_count_set.items(),key=operator.itemgetter(1),reverse=True)    

m=np.array(num)
np.save('num.npy',m)
m=np.array(pos_count_set)
np.save('pos_count_set.npy',m)
m=np.array(neg_count_set)
np.save('neg_count_set.npy',m)
m=np.array(neg_training)
np.save('neg_training.npy',m)
m=np.array(neg_testing)
np.save('neg_testing.npy',m)
m=np.array(pos_training)
np.save('pos_training.npy',m)
m=np.array(pos_testing)
np.save('pos_testing.npy',m)