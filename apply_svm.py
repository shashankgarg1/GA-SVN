# -*- coding: utf-8 -*-

from sklearn import svm

def apply(X,Y,x_test,kernel_bits):
    #made changes to this script to incorporate the curve, addition of y_score and its return is the addition.
#    print "i have entered svm"
    assert len(X)==len(Y)
    option=kernel_bits[0]*2+kernel_bits[1]
    
    if option==0:   #linear svm
#        print "i am in linear mode"
        model=svm.SVC(C=0.95,kernel='linear',tol=1e-3,cache_size=4096)
        model.fit(X,Y)
        predicted= model.predict(x_test)
        y_score=model.decision_function(x_test)
        return predicted,y_score
    elif option==1: #polynomial svm   
#        print "i am in polynomial mode"
        #gam=[1.0/len(X.columns),0.1,0.3,0.5,0.7,0.9]
        gam=1.0/len(X.columns)
#==============================================================================
#         score=-1e9
#         fin_deg=0
#         fin_gam=0
#         for deg in range(2,3):
#         #for deg in range(2,4):
#             for g in gam:
#                 # print deg
#                 # print g
#                 model=svm.SVC(C=0.95,kernel='poly',degree=deg,gamma=g,coef0=1.0,tol=1e-3,cache_size=4096)
#                 model.fit(X,Y)
#                 temp=model.score(X,Y)
#                 if temp>score:
#                     score=temp
#                     fin_deg=deg
#                     fin_gam=g
#         model=svm.SVC(C=0.95,kernel='poly',degree=fin_deg,gamma=fin_gam,coef0=1,tol=1e-3,cache_size=4096)
#==============================================================================
        model=svm.SVC(C=0.95,kernel='poly',degree=2,gamma=gam,coef0=1,tol=1e-3,cache_size=4096)
        model.fit(X,Y)
        predicted=model.predict(x_test)
        y_score=model.decision_function(x_test)
        return predicted,y_score
                    
    elif option==2: #rbf kernel
#        print "i am in rbf mode"
        #gam=[1.0/len(X.columns),0.1,0.3,0.5,0.7,0.9]
        gam=1.0/len(X.columns)
#==============================================================================
#         score=-1e9
#         fin_gam=0
#         for g in gam:
#             model=svm.SVC(C=0.95,kernel='rbf',gamma=g,tol=1e-3,cache_size=4096)
#             model.fit(X,Y)
#             temp=model.score(X,Y)
#             if temp>score:
#                 score=temp
#                 fin_gam=g
#         model=svm.SVC(C=0.95,kernel='rbf',gamma=fin_gam,tol=1e-3,cache_size=4096)
#==============================================================================
        model=svm.SVC(C=0.95,kernel='rbf',gamma=gam,tol=1e-3,cache_size=4096)        
        model.fit(X,Y)
        predicted=model.predict(x_test)
        y_score=model.decision_function(x_test)
        return predicted,y_score
    
    elif option==3: #sigmoid kernel
#        print "i am in sigmoid mode"
        #gam=[1.0/len(X.columns),0.1,0.3,0.5,0.7,0.9]
        gam=1.0/len(X.columns)
#==============================================================================
#         score=-1e9
#         fin_gam=0
#         for g in gam:
#             model=svm.SVC(C=0.95,kernel='sigmoid',gamma=g,coef0=1.0,tol=1e-3,cache_size=4096)
#             model.fit(X,Y)
#             temp=model.score(X,Y)
#             if temp>score:
#                 score=temp
#                 fin_gam=g
#         model=svm.SVC(C=0.95,kernel='sigmoid',gamma=fin_gam,coef0=1.0,tol=1e-3,cache_size=4096)
#==============================================================================
        model=svm.SVC(C=0.95,kernel='sigmoid',gamma=gam,coef0=1.0,tol=1e-3,cache_size=4096)        
        model.fit(X,Y)
        predicted=model.predict(x_test)
        y_score=model.decision_function(x_test)
        return predicted,y_score
    
#==============================================================================
#     model = svm.SVC(kernel='linear', C=1, gamma=1)
#     model.fit(X,Y)
#     model.score(X,Y)
#     #Predict Output
#     predicted= model.predict(x_test)
#     return predicted
#==============================================================================
