import operator
import smart_open
import numpy as np


#-----------------------------导入训练集-----------------------------------------------------------
stopWord=['，','。','《','》','：','；','“','”',"‘","’",'！','？',"'",',',' ','[',']','、','\n']
neg_training=np.load('neg_training.npy')
pos_training=np.load('pos_training.npy')
neg_poem=str(neg_training)
pos_poem=str(pos_training)

#-----------------------------------计算字频-----------------------------------------------------

dic={}
for Char in pos_poem:
    if Char not in stopWord:
        if Char not in dic:
            dic[Char]=1
        else:
            dic[Char]+=1
for Char in neg_poem:
    if Char not in stopWord:
        if Char not in dic:
            dic[Char]=-1
        else:
            dic[Char]-=1
swd = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
#print(swd) #最终训练集版本Pos和Neg文本字数应大致相同

x1=swd[0][1]
x2=swd[-1][1]
x_max=max(abs(x1),abs(x2))

char_count=len(swd)
train_char_score=[] #各个字的情感得分的集合
happiest_char=['喜','乐','欢','欣','笑','悦']
happier_char=['暖','嘉','丰','贺','福','歌','好','爱','兴','迎','娱',]
saddest_char=['哀','泣','悲','愁','怨','忧','恸','恨',]
sadder_char=['悼','忡','惆','怅','惧','寞','闷','憔','悴','泪','叹','伤','苦','凄','痴','嗟','恐','虑','愤','恼','惨',]
for i in range(char_count):
    x=(swd[i][1]+x_max)/(2*x_max)   #将各字的情感分映射到0~1区间
    if swd[i][0] in happiest_char:  #给明显的积极情感字加权
        x=0.51+(x-0.5)*5
        if x>1:
            x=1.0
    elif swd[i][0] in saddest_char:
        x=0.49+(x-0.5)*5
        if x<0:
            x=0.0
    elif swd[i][0] in happier_char:  #给明显的积极情感字加权
        x=0.51+(x-0.5)*2
        if x>1:
            x=1.0
    elif swd[i][0] in sadder_char:
        x=0.49+(x-0.5)*2
        if x<0:
            x=0.0
    train_char_score.append((swd[i][0],x)) 

train_char_score.sort(key=operator.itemgetter(1),reverse=True)
#print(train_char_score)
m=np.array(train_char_score)
np.save("train_char_score.npy",m)#'''