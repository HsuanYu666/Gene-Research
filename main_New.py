# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:37:33 2017

@author: Hsuan Yu 
"""

import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectPercentile
import matplotlib.pyplot as plt
import pickle as pk

    
#%%
#==============================================================================
#  Others Define functions
#==============================================================================
def Confusion_matrix(TestLabel,pred1):
    cm = confusion_matrix(TestLabel,pred1)
#    if float(cm[0][0]+cm[0][1]) >0
    acc1 = float(cm[0][0])/float(cm[0][0]+cm[0][1])
    acc2 = float(cm[1][1])/float(cm[1][0]+cm[1][1])
    acc = (acc1 + acc2) /2        
    return [acc,cm]


 
# Load Features and Label 
f = open("./save_data/Feature_Label","rb")
Feature_Label = pk.load(f)
Feature = Feature_Label[0][:30]
Label_ = Feature_Label[1][:30]
f.close()

# Load Unique gene types
f = open("./save_data/Aa1","rb")
Aa1 = pk.load(f)
f.close()


f = open("fs_idx_31_new","rb")
Idx = pk.load(f)
f.close()

def Train_Val(Feature,Label_):
    kf = KFold(len(Feature),5)
    
    Acc_Train =[]; Acc_Val = []
    for train_index,test_index in kf:
        TrainSet,TestSet = Feature[train_index], Feature[test_index]
        TrainLabel,TestLabel = Label_[train_index], Label_[test_index]
       
    #==============================================================================
    #   Special Idx
    #==============================================================================
        TrainSet = TrainSet[:,Idx]
        TestSet = TestSet[:,Idx]
    #==============================================================================
    #   use best percentile of fs to train again
    #==============================================================================
#        percent = 20 # (i+1)*10
#        fsModel = SelectPercentile(percentile=percent).fit(TrainSet, TrainLabel)
#    
#        TrainSet = fsModel.transform(TrainSet)
#        TestSet = fsModel.transform(TestSet)
    
    #==============================================================================
    #   start training model (compare different algo.)
    #==============================================================================
        svm_model = SVC(kernel = 'linear',class_weight ='balanced' ) 
#        svm_model = LogisticRegression()
#        svm_model = AdaBoostClassifier() 
        svm_model.fit( TrainSet, TrainLabel )
    
        
    #==============================================================================
    #   get predictions
    #==============================================================================
        pred_val = svm_model.predict(TestSet)   
        pred_train = svm_model.predict(TrainSet) 
        
        Acc_Val.append(Confusion_matrix(TestLabel,np.asarray(pred_val))[0])
        Acc_Train.append(Confusion_matrix(TrainLabel,np.asarray(pred_train))[0])
    
    CV_Acc_Val = np.mean(Acc_Val)   
    CV_Acc_Train = np.mean(Acc_Train)  
    
    return [CV_Acc_Val,CV_Acc_Train]
 
   
CV_Acc_Val = []
CV_Acc_Train = []
for i in range(4):
#    print(15+(5*i))
    results = Train_Val(Feature[:15+(5*i)],Label_[:15+(5*i)])
    print("--------Training Size:"+str(15+(5*i))+"-------")
    print("Accuracy of Training:",results[1],"\nAccuracy of Validation:",results[0])
    
    CV_Acc_Val.append(results[0])
    CV_Acc_Train.append(results[1])
        
# plot the learning curve    

Desire_Acc = [0.9]*len(CV_Acc_Val)
fig, ax = plt.subplots()
ax.plot(CV_Acc_Train, 'b-', label='Training')
ax.plot(CV_Acc_Val, 'g-', label='Validation')
ax.plot(Desire_Acc, 'r--', label='Desire')
plt.title("Learning curve  ")
plt.xlabel("Traing Size")
plt.ylabel("Accuracy")
legend = ax.legend(bbox_to_anchor=(1.35, 1.05), shadow=True)