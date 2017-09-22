# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:37:33 2017
@author: Hsuan Yu 
"""
from __future__ import division
from __future__ import print_function

import numpy as np 
import matplotlib.pyplot as plt
import pickle as pk

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectPercentile
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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


def fs_over_cv(feature, label, percent):
    loo = LeaveOneOut(len(feature))
    select_ind = np.array([True]*feature.shape[1])
    for train_index, _ in loo:
        TrainSet = feature[train_index]
        TrainLabel = label[train_index]
        fsModel = SelectPercentile(percentile=percent).fit(TrainSet, TrainLabel)
        select_ind = np.logical_and(select_ind, fsModel.get_support())
    return np.where(select_ind==True)[0]


def validate(feature, label, clf, percent=100):
    result = np.array(label)
    loo = LeaveOneOut(len(feature))
    for train_index, test_index in loo:
        TrainSet, TestSet = feature[train_index], feature[test_index]
        TrainLabel = label[train_index]
        
        fsModel = SelectPercentile(percentile=percent).fit(TrainSet, TrainLabel)
        TrainSet = fsModel.transform(TrainSet)
        TestSet = fsModel.transform(TestSet)
        
        clf.fit(TrainSet, TrainLabel)
        result[test_index] = clf.predict(TestSet)
    return Confusion_matrix(label, result)[0]


def best_percent(feature, label, clf):
    percents = np.arange(10, 101, 10)
    accuracies = []
    for p in percents:
        accuracies.append(validate(feature, label, clf, p))
        
    return percents[np.argmax(accuracies)]


def Train_Val(Feature, Label_, model):
    loo = LeaveOneOut(len(Feature))
    Acc_Train = []
    result = np.array(Label_)
    for train_index, test_index in loo:
        TrainSet, TestSet = Feature[train_index], Feature[test_index]
        TrainLabel = Label_[train_index]
        
        p = 30#best_percent(TrainSet, TrainLabel, model)
        fs_ind = fs_over_cv(TrainSet, TrainLabel, p)
        TrainSet, TestSet = TrainSet[:, fs_ind], TestSet[:, fs_ind]
        
        model.fit(TrainSet, TrainLabel)
        
        pred_val = model.predict(TestSet)   
        pred_train = model.predict(TrainSet) 
        result[test_index] = pred_val
        Acc_Train.append(Confusion_matrix(TrainLabel, np.asarray(pred_train))[0])
    
    CV_Acc_Val =  Confusion_matrix(Label_, result)[0] 
    CV_Acc_Train = np.mean(Acc_Train)  
    
    return CV_Acc_Val, CV_Acc_Train

#%%
models = [KNeighborsClassifier(2),
          SVC(C=10, kernel='linear', class_weight='balanced'),
          LogisticRegression(C=10, class_weight='balanced'),
          GaussianNB()]

if __name__ == '__main__':
    # Load Features and Label and fs_Index
    Feature_Label = pk.load(open("./save_data/Feature_Label","rb"))
#    Idx1 = pk.load(open("./save_data/Idx_Backward","rb"))
#    Idx2 = pk.load(open("./save_data/Idx_Forward","rb"))
#    Idx = [val for val in Idx1 if val in Idx2]             # 6 genes
    Idx = pk.load(open("./save_data/fs_idx_31_new","rb"))   # 48 genes
    
    # Check what's gene names
#    All_Unique_Genes = pk.load(open("./save_data/Aa1","rb"))
#    print(All_Unique_Genes[Idx])

    Feature = Feature_Label[0]
    Label_ = Feature_Label[1]
    
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

