#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import _pickle as cPickle
from datetime import datetime
import operator
import yaml
from sklearn import preprocessing

with open('reviews_Beauty_5.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile, lines=True)
  
df= df.sort_values(by=['unixReviewTime'], ascending=True)
print(df.head())

first = int(df.shape[0]*0.8)
last= int(df.shape[0]*0.1)
df_train= df.head(first).reset_index(drop=True)
df_test= df.tail(last).reset_index(drop=True)

item_seq_train=df_train.groupby("reviewerID")['asin'].apply(list).values
item_seq_test=df_test.groupby("reviewerID")['asin'].apply(list).values

print(len(item_seq_train), len(item_seq_test))

items= df.asin.unique()
print(len(items))
users= df.reviewerID.unique()
print(len(users))
np.save("vocab_amazon",items)

def input_sequences(new_list):
    input_seq=[]
    target=[]
    user_input=[]
    for i in range(0,len(new_list)):
        seq_len = len(new_list[i])
        for j in range(1,seq_len):
            input_seq.append(new_list[i][:j])
            target.append(new_list[i][j])
        
    return input_seq,target

train_x,target_train=input_sequences(item_seq_train) 

test_x,target_test=input_sequences(item_seq_test)

#save in appropriate folders
with open('amazon_train_x','wb') as f:
    cPickle.dump(train_x,f)
with open('amazon_train_y','wb') as f:
    cPickle.dump(target_train,f)
with open('amazon_test_y','wb') as f:
    cPickle.dump(target_test,f)
with open('amazon_test_x','wb') as f:
    cPickle.dump(test_x,f)
with open('all_train_seq.txt','wb') as f:
  cPickle.dump(item_seq_train, f)
  
