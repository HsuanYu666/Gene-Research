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
	
	# Load test data
    Feature_test = pk.load(open("./save_data/Mutant_allele_20171026","rb"))
    Feature_test = Feature_test[:, Idx]
	
	for clf in models:
		clf.fit(Feature, Label_)
		predict = clf.predict(Feature_test)
		print(predict)

