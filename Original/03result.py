import gensim
from gensim.models import KeyedVectors
import numpy as np
from gensim.models import Word2Vec
import smart_open
#-----------------------------------导入情感字训练集---------------------------------
def loadList(addr):
    a=np.load(addr)
    lis=a.tolist()
    return lis
train_char_score=loadList('train_char_score.npy')
train_char_dict=dict(train_char_score)

#print(train_char_dict)
#------------------------------------导入字向量部分----------------------------------
word2vec_model = KeyedVectors.load('word2vec_model')

word2vec_dict={}
for word, vector in zip(word2vec_model.vocab,word2vec_model.vectors):
    word2vec_dict[word]=vector
    #print(word)

#word2vec_dict={'话',[0,0,0,...]}
#全部字向量【字典】导入
    
#-----------------------------------导入测试集部分--------------------------------
a=np.load('neg_testing.npy')
neg_testing=a.tolist()
a=np.load('pos_testing.npy')
pos_testing=a.tolist()
#-------------------------测试部分---------------------------------------------

#-------------------------定义函数---------------------------------------------
print('--------------------------------------------------------------')

score=0
relNum=5
poemValue=0
charLen=0
emoValue=0   #全诗未平均前的情感总得分
charValue=[]
emoDe=0.5      #确定积极&消极的分界点
stopWord=['，','。','《','》','：','；','“','”',"‘","’",'！','？','?']

def getSimilarCount(Char):
    try:
        test_char_vec=word2vec_dict[Char]
    except KeyError:
        return float(0.5)
    s=word2vec_model.most_similar(positive=[test_char_vec],topn=relNum)
    total_emoScore=0
    j=0
    for similarChar in range(relNum):
         relativity=s[similarChar][1]                            #猜测字与所选字的相关度
         try:
             emoScore=float(train_char_dict[s[similarChar][0]])  #单个猜测字的情感得分
             total_emoScore+=relativity*emoScore             
             j+=relativity
         except KeyError:
             pass
    if j!=0:                     
        ave_value=total_emoScore/j                 #得到该字的最终情感分
        return ave_value
    else:
        return float(0.5)
        #charValue.append(Char+'：'+str(ave_value))

def SentiAnalysis(poem):
    charLen=0
    emoValue=0
    for Char in poem:
        if Char in train_char_dict:
            ave_value=float(train_char_dict[Char])
        elif Char not in stopWord:
            ave_value=getSimilarCount(Char)
            #print(Char + ":"+str(ave_value))
        
        if ave_value !=0.5:
            charLen+=1
            emoValue+=ave_value
    if charLen==0:
        return float(0.5)
    else:
        result=emoValue/charLen
    return result

#----------------------测试结果----------------------------------------------

right=0
total=0
for i in range(len(pos_testing)): 
    poemValue=SentiAnalysis(pos_testing[i])
    #print(poemValue)
    if poemValue>emoDe:
        right+=1
    #else:
       #print(poemValue)
    total+=1
'''print("score="+str(right/total))
print("---------------------------------------------------")'''

total_right=right
total_total=total
right=0
total=0
for i in range(len(neg_testing)): 
    poemValue=SentiAnalysis(neg_testing[i])
    #print(poemValue)
    if poemValue<emoDe:
        right+=1
    #else:
        #print(neg_testing[i],poemValue)
    total+=1
'''
print("---------------------------------------------------")
print("score="+str(right/total))
'''
total_right+=right
total_total+=total
print("---------------------------------------------------")
print("total score="+str(total_right/total_total))


'''testText="独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。 "
result=SentiAnalysis(testText)
print(testText)
print(result)#'''