# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import  preprocessing

def read():
    # import apply_svm
#    dataset=pd.read_table('/home/shashank/Desktop/WDBC.dat',sep=',',header=None)
    
    
    dataset=pd.read_table('/home/shashank/Desktop/WDBC.dat',sep=',',header=None)


    #removing identifier
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    preprocessed_dataset=dataset.drop(0,1)
    
#    sh=preprocessed_dataset.shape
#    ind=[False]*sh[0]
#    for i in range(sh[0]):
#        for j in range(1,sh[1]):
#            if preprocessed_dataset.ix[i,j]=='?':
#                ind[i]=True
#    preprocessed_dataset.drop(preprocessed_dataset.index[ind],inplace=True)
#    preprocessed_dataset.reset_index(drop=True,inplace=True)
   
   # 2=benign,4= malignant
#    sh=preprocessed_dataset.shape
#    for i in range(sh[0]):
#        if preprocessed_dataset.ix[i,10]==2:
#            preprocessed_dataset.ix[i,10]='B'
#        else:
#            preprocessed_dataset.ix[i,10]='M'
#    
    features_no=len(preprocessed_dataset.columns)-1
    
    X=preprocessed_dataset.ix[:,2:]
    Y=preprocessed_dataset.ix[:,1]
    
#    X=preprocessed_dataset.ix[:,1:9]    
#    Y=preprocessed_dataset.ix[:,10]
    
    X_preprocessed=preprocessing.scale(X)
    X_preprocessed=pd.DataFrame(X_preprocessed)
    return [X_preprocessed,Y,features_no]
    # apply_svm.apply(preprocessed_dataset.ix[:,2:31],preprocessed_dataset.ix[:,1],preprocessed_dataset.ix[119:120,2:31])