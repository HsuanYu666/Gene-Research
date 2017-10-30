# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:37:33 2017
@author: Hsuan Yu 
"""
from __future__ import division
from __future__ import print_function

import numpy as np 
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

def Confusion_matrix(TestLabel, pred1):
    cm = confusion_matrix(TestLabel, pred1)
    acc = []
    for ind, c in enumerate(cm):
        if np.sum(c) == 0:
            acc.append(0)
        else:
            acc.append(c[ind]/ np.sum(c))
    
    return np.mean(acc), cm

models = [KNeighborsClassifier(5),
          SVC(C=1, kernel='linear', class_weight='balanced'),
          LogisticRegression(C=0.01, class_weight='balanced'),
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
	
	# Load test data
    Feature_test = pk.load(open("./save_data/Mutant_allele_20171026","rb"))
    Feature_test = Feature_test[:40][:, Idx]
    Label_test = np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,
                           0,0,1,0,0,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1])
	
    for clf in models:
      clf.fit(Feature, Label_)
      predict = clf.predict(Feature_test)
      print(Confusion_matrix(Label_test, predict)[0])
      print(Confusion_matrix(Label_test, predict)[1])
