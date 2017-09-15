# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:37:33 2017
@author: Hsuan Yu 
"""
from __future__ import division
from __future__ import print_function

import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectPercentile
import matplotlib.pyplot as plt
import pickle as pk
from sklearn.naive_bayes import GaussianNB


from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#%%
#==============================================================================
#  Others Define functions
#==============================================================================
def Confusion_matrix(TestLabel, pred1):
    cm = confusion_matrix(TestLabel, pred1)
    acc = []
    for ind, c in enumerate(cm):
        if np.sum(c) == 0:
            acc.append(0)
        else:
            acc.append(c[ind]/ np.sum(c))
    
    return np.mean(acc), cm

def Train_Val(Feature, Label_,model):
    loo = LeaveOneOut(len(Feature))
    Acc_Train, Acc_Val = [], []
    
    for train_index, test_index in loo:
        TrainSet, TestSet = Feature[train_index], Feature[test_index]
        TrainLabel, TestLabel = Label_[train_index], Label_[test_index]
    
    #==============================================================================
    #   use best percentile of fs to train again
    #==============================================================================
#        percent = 20 # (i+1)*10
#        fsModel = SelectPercentile(percentile=percent).fit(TrainSet, TrainLabel)
#    
#        TrainSet = fsModel.transform(TrainSet)
#        TestSet = fsModel.transform(TestSet)
    
    #==============================================================================
    #   start training model 
    #==============================================================================
        model.fit(TrainSet, TrainLabel)
    
    #==============================================================================
    #   get predictions
    #==============================================================================
        pred_val = model.predict(TestSet)   
        pred_train = model.predict(TrainSet) 
        
        Acc_Val.append(Confusion_matrix(TestLabel, np.asarray(pred_val))[0])
        Acc_Train.append(Confusion_matrix(TrainLabel, np.asarray(pred_train))[0])
    
    CV_Acc_Val = np.mean(Acc_Val)   
    CV_Acc_Train = np.mean(Acc_Train)  
    
    return CV_Acc_Val, CV_Acc_Train

    
models = [KNeighborsClassifier(3),
          SVC(C=10, kernel='linear', class_weight='balanced'),
          LogisticRegression(C=10, class_weight='balanced'),
          GaussianNB()]

if __name__ == '__main__':
    # Load Features and Label and fs_Index
    Feature_Label = pk.load(open("./save_data/Feature_Label","rb"))
    Idx1 = pk.load(open("./save_data/Idx_Backward","rb")) 
    Idx2 = pk.load(open("./save_data/Idx_Forward","rb"))   
    
#    Idx = [val for val in Idx1 if val in Idx2]            # 6 genes 
    Idx = pk.load(open("./save_data/fs_idx_31_new","rb"))   # 48 genes
        
    # Check what's gene names
#    All_Unique_Genes = pk.load(open("./save_data/Aa1","rb"))
#    print(All_Unique_Genes[Idx])

    Feature = Feature_Label[0][:31][:, Idx]
    Label_ = Feature_Label[1][:31]
    num = 1 
    for model in models:
        print("============ Model"+str(num)+" ============")
        results = Train_Val(Feature,Label_,model)
        print("Accuracy of Training:",results[1],"\nAccuracy of Validation:",results[0])
        num += 1
        
        
    # Learning Curve
#        CV_Acc_Val, CV_Acc_Train = [], []
#        for i in range(4):
#            results = Train_Val(Feature[:15+(5*i)], Label_[:15+(5*i)],model)
#            print("--------Training Size:"+str(15+(5*i))+"-------")
#            print("Accuracy of Training:",results[1],"\nAccuracy of Validation:",results[0])      
#            CV_Acc_Val.append(results[0])
#            CV_Acc_Train.append(results[1])        
#        # plot the learning curve    
#        Desire_Acc = [0.9]*len(CV_Acc_Val)
#        fig, ax = plt.subplots()
#        ax.plot(CV_Acc_Train, 'b-', label='Training')
#        ax.plot(CV_Acc_Val, 'g-', label='Validation')
#        ax.plot(Desire_Acc, 'r--', label='Desire')
#        plt.title("Learning curve  ")
#        plt.xlabel("Training Size")
#        plt.ylabel("Accuracy")
#        legend = ax.legend(bbox_to_anchor=(1.35, 1.05), shadow=True)
#        plt.show()

