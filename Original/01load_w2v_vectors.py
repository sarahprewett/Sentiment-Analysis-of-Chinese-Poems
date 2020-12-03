# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:17:14 2019

@author: Administrator
"""
import gensim
from gensim.models import KeyedVectors
import numpy as np
from gensim.models import Word2Vec
import smart_open
from sklearn.model_selection import train_test_split

#---------------------------导入word2vector模型，只需执行一次-------------
#file = 'D:/02英才计划/课题后期/测试过程/sgns.sikuquanshu.word/sgns.sikuquanshu.word'#词向量的位置
'''file = 'D:/02英才计划/课题后期/SentimentAnalysis/Original/sgns.sikuquanshu.bigram'
#file = r'D:/02英才计划/课题后期/测试过程/ppmi.sikuquanshu.word/ppmi.sikuquanshu.word'
#file = 'D:/02英才计划/课题后期/测试过程/ppmi.sikuquanshu.bigram/ppmi.sikuquanshu.bigram'
loaded_model = KeyedVectors.load_word2vec_format(file, binary=False)
word2vec_model=loaded_model.wv
del loaded_model
word2vec_model.save('D:/02英才计划/课题后期/SentimentAnalysis/Original/word2vec_model')'''

#----------------------------导入训练、测试集，可执行多次----------------                     
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

m=np.array(neg_training)
np.save('neg_training.npy',m)
m=np.array(neg_testing)
np.save('neg_testing.npy',m)
m=np.array(pos_training)
np.save('pos_training.npy',m)
m=np.array(pos_testing)
np.save('pos_testing.npy',m)
