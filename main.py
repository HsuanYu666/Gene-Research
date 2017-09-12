# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:37:33 2017

@author: HsuanYu
"""

import numpy as np 
import pandas as pd
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectPercentile
import matplotlib.pyplot as plt
import pickle as pk



#%%
#==============================================================================
#  Define functions
#==============================================================================
def Confusion_matrix(TestLabel,pred1):
    cm = confusion_matrix(TestLabel,pred1)
#            acc = float(cm[0][0]+cm[1][1])/len(TestLabel)
    acc1 = float(cm[0][0])/float(cm[0][0]+cm[0][1])
    acc2 = float(cm[1][1])/float(cm[1][0]+cm[1][1])
    acc = (acc1 + acc2) /2        
    return [acc,cm]


#==============================================================================
# Load Label & Features & Unique Gene Names
#==============================================================================

# Unique Gene Names
f = open("./save_data/Aa1","rb")
Aa1 = pk.load(f)
f.close()
    
# Label & Features
f = open("./save_data/People31_Feature_Label","rb")
People31_Feature_Label = pk.load(f)
Feature = People31_Feature_Label[0]
Label_ = People31_Feature_Label[1]
f.close()

f = open("fs_idx_31_new","rb")
Idx = pk.load(f)
f.close()

kf = KFold(len(Feature),len(Feature))
Fs_Result = []  ; Fs_Result2= [] ; Fs_Idx = []
for i in range(1):
    Pred = []
    for train_index,test_index in kf:
        TrainSet,TestSet = Feature[train_index], Feature[test_index]
        TrainLabel,TestLabel = Label_[train_index], Label_[test_index]
       
    #==============================================================================
    #         # Special Idx
#    #==============================================================================
        TrainSet = TrainSet[:,Idx]
        TestSet = TestSet[:,Idx]
    #==============================================================================
    #       ## use best percentile of fs to train again
    #==============================================================================
#        percent = 10 #(i+1)*10
#        fsModel = SelectPercentile(percentile=percent).fit(TrainSet, TrainLabel)
#    
#        TrainSet = fsModel.transform(TrainSet)
#        TestSet = fsModel.transform(TestSet)

    #==============================================================================
    #        ## start training model  
    #==============================================================================
        svm_model = SVC(kernel = 'linear',class_weight ='balanced' )        
        svm_model.fit( TrainSet, TrainLabel )
        
    #==============================================================================
    #         ## get predictionss
    #==============================================================================
        pred = svm_model.predict(TestSet) # SVC
        Pred.extend(pred)            
        
#        boool = fsModel.get_support()
#        Fs_Idx.append(boool)
    Aaa1 = Confusion_matrix(Label_,np.asarray(Pred))
    
    Fs_Result.append(Aaa1[0])
    Fs_Result2.append(Aaa1[1])
    
            
print(max(Fs_Result))

if len(Fs_Result)>1:
    plt.title("Accuracy of Each Idx")
    plt.plot(Fs_Result)
    plt.xlabel("Index")
    plt.ylabel("Accuracy")
