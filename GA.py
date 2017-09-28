# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import apply_svm

class genetic_algo:
    
    def __init__(self,features,X,Y,population_no=500,generations=150,crossover_prob=0.05,mutation_prob=0.001,k_crossover=0.05,k_mutation=0.01,partition=0.70, cross_validation=False, fold_cv=1):
        self.population=np.zeros(shape=(population_no,features+2),dtype=bool) #+2 is for kernel bits == linear, polynomial, rbf, sigmoid
        self.X=X
        self.Y=Y
        self.generations=generations
        self.distinct_chromosomes=[]
        self.hash=[] #mapping population to distinct chromosomes
        self.chromosome_count=[]
        self.new_population=[]
        self.crossover_prob=crossover_prob
        self.mutation_prob=mutation_prob
        self.k_crossover=k_crossover
        self.k_mutation=k_mutation
        self.best_chromosome=[]
        self.partition=partition
        self.cross_validation=cross_validation
        self.fold_cv=fold_cv
        
    def initialise_population(self):
        rows=len(self.population)
        cols=len(self.population[0])
        for i in range(rows):
            for j in range(cols):
                self.population[i,j]=random.choice([True,False])
       #  print "going out from initialise_population"
    
    def mapandhash(self):
        k=0
        self.distinct_chromosomes=[]
        self.hash=[]
        self.chromosome_count=[]
        for i in range(len(self.population)):
            flag=0
            for k in range(len(self.distinct_chromosomes)):
                if np.all(self.population[i]==self.distinct_chromosomes[k]):
                    self.hash.append((i,k))
                    self.chromosome_count[k]=self.chromosome_count[k]+1
                    flag=1
                    break
            if flag==0:
                self.distinct_chromosomes.append(self.population[i])
                self.hash.append((i,k+1))
                self.chromosome_count.append(1)
        (x,y)=self.hash[0]
        self.hash[0]=(x,y-1)
#        print "going out from mapandhash"
    
    def fitness(self,TP,FP,FN,TN):
        P=TP+FN
        N=FP+TN
        accuracy=(TP+TN)/(P+N)
        
        assert N>0
        assert P>0
        '''if N==0:
            specificity=float("inf")
        else:'''
        specificity=1.0*TN/N
        
        '''if P==0:
            sensitivity=float("inf")
        else:'''
        sensitivity=1.0*TP/P
        score=(math.sqrt(accuracy**2+specificity**2+sensitivity**2))/3.0+0.000001
        assert score>=0
        return score
    
    def build_solution(self):
        self.initialise_population()       
            
        for gen in range(self.generations):
            self.mapandhash()
            score=[]
            total_fitness=0
            for i in range(len(self.distinct_chromosomes)):
                length=len(self.distinct_chromosomes[0])
                X=self.X.iloc[:,self.distinct_chromosomes[i][0:length-2]]
                
                # print X                
                
                mid=int(round(self.partition*len(X),0))
                predicted=apply_svm.apply(X.ix[0:mid-1,:],self.Y[0:mid],X.ix[mid:,:],self.distinct_chromosomes[i][length-2:])
                
#                print "successfully returned from apply_svm"                
                
                y_test=self.Y[mid:]
                y_test=list(y_test)
                predicted=list(predicted)
                # print type(predicted)
                assert len(y_test)==len(predicted)
                
                TP=0
                FP=0
                FN=0
                TN=0
                for j in range(len(predicted)):
                    if y_test[j]=='B' and predicted[j]=='B':
                        TP=TP+1
                    elif y_test[j]=='B' and predicted[j]=='M':
                        FN=FN+1
                    elif y_test[j]=='M' and predicted[j]=='B':
                        FP=FP+1
                    elif y_test[j]=='M' and predicted[j]=='M':
                        TN=TN+1
                    else:
                        print 'Error'
                        exit()
                score.append(self.fitness(TP,FP,FN,TN))
                total_fitness=total_fitness+score[i]
            normalised_fitness=[]
            for i in range(len(score)):
                normalised_fitness.append(score[i]/total_fitness)
            self.reproduction2(normalised_fitness)
            
#            print "successfully returned from reproduction"            
            
            check_crossover=random.random()
            if check_crossover<=self.crossover_prob:
                print "going into crossover"                
                self.crossover()
#                print "successfully returned from crossover"
                
            check_mutation=random.random()
            if check_mutation<=self.mutation_prob:
                print "going into mutation"
                self.mutation()
#                print "successfully returned from mutation"
            if gen<self.generations-1:
                self.population=self.new_population
        
        return score     
    
    
    def build_solution2(self):
        self.initialise_population()
        sol=[]
        if self.cross_validation==True:
            self.partition=1.0*(self.fold_cv-1)/self.fold_cv
        X=self.X
        Y=self.Y
        for folds in range(self.fold_cv):
#            print folds
            mid=int(round(self.partition*len(X),0))
#            print len(X)
            assert len(X)==len(self.X)
            assert len(Y)==len(self.Y)
            train_X=X.iloc[0:mid,:]
            train_Y=Y[0:mid]
            assert len(train_X)==len(train_Y)
            test_X=X.iloc[mid:,:]
            test_Y=Y[mid:]
            assert len(test_X)==len(test_Y)
            
            for gen in range(self.generations):
                self.mapandhash()
                score=[]
                total_fitness=0
                for i in range(len(self.distinct_chromosomes)):
                    length=len(self.distinct_chromosomes[0])
                    train_gen_X=train_X.iloc[:,self.distinct_chromosomes[i][0:length-2]]
                    test_gen_X=test_X.iloc[:,self.distinct_chromosomes[i][0:length-2]]
#                   print len(X)
                    
                    
                    if len(train_gen_X.columns)==0:
                        score.append(0)
                        continue
                    
                    
                    predicted,_=apply_svm.apply(train_gen_X,train_Y,test_gen_X,self.distinct_chromosomes[i][length-2:])
                    
#                   print "successfully returned from apply_svm"                
                    
                    y_test=list(test_Y)
                    predicted=list(predicted)
                    # print type(predicted)
                    assert len(y_test)==len(predicted)
                    
                    TP=0
                    FP=0
                    FN=0
                    TN=0
                    for j in range(len(predicted)):
                        if y_test[j]=='B' and predicted[j]=='B':
                            TP=TP+1
                        elif y_test[j]=='B' and predicted[j]=='M':
                            FN=FN+1
                        elif y_test[j]=='M' and predicted[j]=='B':
                            FP=FP+1
                        elif y_test[j]=='M' and predicted[j]=='M':
                            TN=TN+1
                        else:
                            print 'Error'
                            exit()
                    score.append(self.fitness(TP,FP,FN,TN))
                    total_fitness=total_fitness+score[i]
                normalised_fitness=[]
                for i in range(len(score)):
                    normalised_fitness.append(score[i]/total_fitness)
                self.reproduction2(normalised_fitness)
                
#               print "successfully returned from reproduction"            
                
                check_crossover=random.random()
                if check_crossover<=self.crossover_prob:
 #                   print "going into crossover"                
                    self.crossover()
#                   print "successfully returned from crossover"
                    
                check_mutation=random.random()
                if check_mutation<=self.mutation_prob:
#                    print "going into mutation"
                    self.mutation()
#                   print "successfully returned from mutation"
                if gen<self.generations-1:
                    self.population=self.new_population
            
            sol.append(self.final_solution2(score,train_X,train_Y,test_X,test_Y))
            X=test_X.append(train_X,ignore_index=True)
            #print len(train_X)
            Y=test_Y.append(train_Y)
            #print len(Y)
        return sol   
        
    def final_solution2(self,score,train_X,train_Y,test_X,test_Y):
        max_score=max(score)
        index=score.index(max_score)
        self.best_chromosome=self.distinct_chromosomes[index]
#        print self.best_chromosome
        
        length=len(self.best_chromosome)
        train_X=train_X.iloc[:,self.best_chromosome[0:length-2]]
        test_X=test_X.iloc[:,self.best_chromosome[0:length-2]]
        
        #made changes to incorporate the curve, the same has been done in apply_svm, addition of y_score and its return is the change for this script.
        predicted,y_score=apply_svm.apply(train_X,train_Y,test_X,self.best_chromosome[length-2:])
        y_test=test_Y
        y_test=list(y_test)
        predicted=list(predicted)
        # print type(predicted)
        assert len(y_test)==len(predicted)
        
        TP=0
        FP=0
        FN=0
        TN=0
        for j in range(len(predicted)):
            if y_test[j]=='B' and predicted[j]=='B':
                TP=TP+1
            elif y_test[j]=='B' and predicted[j]=='M':
                FN=FN+1
            elif y_test[j]=='M' and predicted[j]=='B':
                FP=FP+1
            elif y_test[j]=='M' and predicted[j]=='M':
                TN=TN+1
            else:
                print 'Error in malignant and benign'
                exit()
        precision=(1.0*TP)/(TP+FP)
        recall=(1.0*TP)/(TP+FN)
        specificity=(1.0*TN)/(TN+FP)
        f1score=(2.0*precision*recall)/(precision+recall)
        accuracy=(1.0*(TP+TN))/(TP+TN+FP+FN)
        print "Precision is", precision
        print "Recall/Sensitivity is", recall
        print "F1 Score is", f1score
        print "Accuracy is", accuracy
        print "Specificity is", specificity
        return (precision,recall,f1score,accuracy,specificity,TP,TN,FP,FN,y_test,y_score)
        
        
    def final_solution(self,score):
        max_score=max(score)
        index=score.index(max_score)
        self.best_chromosome=self.distinct_chromosomes[index]
#        print self.best_chromosome
        
        length=len(self.best_chromosome)
        X=self.X.iloc[:,self.best_chromosome[0:length-2]]
        mid=int(round(self.partition*len(X),0))        
        
        predicted=apply_svm.apply(X.ix[0:mid-1,:],self.Y[0:mid],X.ix[mid:,:],self.best_chromosome[length-2:])
        y_test=self.Y[mid:]
        y_test=list(y_test)
        predicted=list(predicted)
        # print type(predicted)
        assert len(y_test)==len(predicted)
        
        TP=0
        FP=0
        FN=0
        TN=0
        for j in range(len(predicted)):
            if y_test[j]=='B' and predicted[j]=='B':
                TP=TP+1
            elif y_test[j]=='B' and predicted[j]=='M':
                FN=FN+1
            elif y_test[j]=='M' and predicted[j]=='B':
                FP=FP+1
            elif y_test[j]=='M' and predicted[j]=='M':
                TN=TN+1
            else:
                print 'Error in malignant and benign'
                exit()
        precision=(1.0*TP)/(TP+FP)
        recall=(1.0*TP)/(TP+FN)
        f1score=(2.0*precision*recall)/(precision+recall)
        accuracy=(1.0*(TP+TN))/(TP+TN+FP+FN)
        print "Precision is", precision
        print "Recall is", recall
        print "F1 Score is", f1score
        print "Accuracy is", accuracy
        
        
    def reproduction2(self,normalised_fitness):
        assert len(normalised_fitness)==len(self.distinct_chromosomes)
        roulette=[]
        roulette.append(normalised_fitness[0])
        for i in range(1,len(normalised_fitness)):
            roulette.append(roulette[i-1]+normalised_fitness[i])
        assert len(roulette)==len(self.distinct_chromosomes)
        rand=[]
        for i in range(len(self.population)):
            rand.append(random.random())
        assert len(rand)==len(self.population)
        self.new_population=[]
        for i in range(len(rand)):
            flag=0
            for j in range(len(roulette)):
                # print rand[i], roulette[j]
                if rand[i]<=roulette[j]:
                    self.new_population.append(self.distinct_chromosomes[j])
                    flag=1
                    break
            if flag==0:
                print "error in roulette logic"
                exit()
        self.new_population=np.array(self.new_population)
        
        assert len(self.new_population)==len(self.population)
        assert len(self.new_population[0])==len(self.population[0])
    
    def reproduction(self,normalised_fitness):
        assert len(normalised_fitness)==len(self.distinct_chromosomes)
        
        
        expected_count=[x*len(self.population) for x in normalised_fitness]
        expected_count=[int(math.ceil(x)) for x in expected_count]
        
#        print [x for x in expected_count]
#        print [x for x in self.chromosome_count]
        roulette=[]
        roulette.append(expected_count[0])
        for i in range(1,len(expected_count)):
            roulette.append(roulette[i-1]+expected_count[i])
        
        assert len(roulette)==len(self.distinct_chromosomes)
        
        rand=[]
        for i in range(len(self.population)):
            rand.append(random.randint(0,len(self.population)-1))
        
        assert len(rand)==len(self.population)
        #assert roulette[len(roulette)-1]==len(self.population)-1
        
#        print roulette[len(roulette)-1]
        
        self.new_population=[]
        for i in range(len(rand)):
            flag=0
            for j in range(len(roulette)):
                # print rand[i], roulette[j]
                if rand[i]<=roulette[j]:
                    self.new_population.append(self.distinct_chromosomes[j])
                    flag=1
                    break
            if flag==0:
                print "error in roulette logic"
                exit()
        self.new_population=np.array(self.new_population)
        
        assert len(self.new_population)==len(self.population)
        assert len(self.new_population[0])==len(self.population[0])
    
    def crossover(self):
        no=int(round(self.k_crossover*len(self.population),0))
        if no%2==1:
            no=no+1
        
        rand=random.sample(range(len(self.population)),no)
        
        for i in range(no/2):
            index_first_chromosome=rand[2*i]
            index_second_chromosome=rand[2*i+1]
            
            position=random.randint(0,len(self.new_population[0])-1)
            x=self.new_population[index_first_chromosome][position]
            y=self.new_population[index_second_chromosome][position]
            
            self.new_population[index_first_chromosome][position]=y
            self.new_population[index_second_chromosome][position]=x
    
    def mutation(self):
        no=int(round(self.k_mutation*len(self.population),0))
        rand=random.sample(range(len(self.population)),no)
        for i in range(no):
            index=rand[i]
            position=random.randint(0,len(self.new_population[0])-1)
            assert self.new_population[index][position]==True or self.new_population[index][position]==False
            if self.new_population[index][position]==True:
                self.new_population[index][position]=False
            else:
                self.new_population[index][position]=True
                