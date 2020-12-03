# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:13:20 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:01:54 2019

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 18:00:21 2019

@author: Administrator
"""
from sklearn.model_selection import train_test_split
import operator
import smart_open
import numpy as np
import math
from gensim.models import KeyedVectors
#---------------------------读取数据------------------------------------------------
#stopWord=['乎','也','兮','之','而','其','，','。','《','》','：','；','“','”',"‘","’",'！'] #注意：添加标点符号会降低正确率，原因未知；加上“之乎者也”对正确率的影响不明显。
stopWord=['，','。','《','》','：','；','“','”',"‘","’",'！']

def loadList(addr):
    a=np.load(addr)
    lis=a.tolist()
    return lis
neg_testing=loadList('neg_testing.npy')
pos_testing=loadList('pos_testing.npy')
#pos_testing="花好月圆"
#---------------------------训练数据------------------------------------------------
pos_count_list=loadList('pos_count_set.npy')
neg_count_list=loadList('neg_count_set.npy')
pos_training=loadList('pos_training.npy')
neg_training=loadList('neg_training.npy')
pos_count_set=dict(pos_count_list)
neg_count_set=dict(neg_count_list)
num=loadList('num.npy')

pos_num=num[0]
neg_num=num[1]

#---------------------------测试结果------------------------------------------------

right=0
total=0

def sentiAnalysis(poemQ): #poemQ为测试诗集
    result=[]
    for j in range(len(poemQ)):  #对于测试集中的每一首诗
        wordCount=0
        pos_prob_list=[]   
        neg_prob_list=[]    
        for word in poemQ[j]:   #“春”
            if word not in stopWord:
                wordCount+=1
                try:
                    posFreq=pos_count_set[word]
                except KeyError:
                    posFreq=0
                try:
                    negFreq=neg_count_set[word]
                except KeyError:
                    negFreq=0
                pos_prob_list.append((posFreq+1)/(pos_num+wordCount))
                neg_prob_list.append((negFreq+1)/(neg_num+wordCount))
                
        pos_prob=1
        for i in pos_prob_list:
            pos_prob*=i
          
        neg_prob=1
        for i in neg_prob_list:
            neg_prob*=i
  
        if pos_prob>neg_prob:
            answer=1
        elif pos_prob<neg_prob:
            answer=0
            #print(poemQ[j])
        else:
            answer=0.5
        result.append(answer)
    return result

posResult=sentiAnalysis(pos_testing)
negResult=sentiAnalysis(neg_testing)
#print(posResult)

def score(result,source):
    total=len(result)
    right=0
    if source==True:
        for poemRes in result:
            if poemRes==1:
                right+=1
    if source==False:
        for poemRes in result:
            if poemRes==0:
                right+=1
    return right/total

posScore=score(posResult,True)
negScore=score(negResult,False)

print((posScore+negScore)/2)

#print(testRes)'''