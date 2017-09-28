# -*- coding: utf-8 -*-
import read_data
import GA
import plots
#made changes to this script to incorporate the curve, addition of plots after the print avg specificity is the change.

[X,Y,features_no]=read_data.read()

obj=GA.genetic_algo(population_no=1000,features=features_no,X=X,Y=Y,generations=1000,crossover_prob=.05,mutation_prob=.001,partition=0.70, cross_validation=False, fold_cv=1)
sol=obj.build_solution2()
#score=obj.build_solution()
#obj.final_solution(score)
precision=0
recall=0
f1score=0
accuracy=0
specificity=0
for i in range(len(sol)):
    precision=precision+sol[i][0]
    recall=recall+sol[i][1]
    f1score=f1score+sol[i][2]
    accuracy=accuracy+sol[i][3]
    specificity=specificity+sol[i][4]

tot=len(sol)
print "avg precision", precision/tot
print "avg recall", recall/tot
print "avg f1score", f1score/tot
print "avg accuracy", accuracy/tot
print "avg specificity",specificity/tot
#
#plots.plot_graph(sol)
#
import pickle
with open('sol_70_30', 'wb') as output:
    pickle.dump(sol, output, pickle.HIGHEST_PROTOCOL)

with open('sol_50_50','rb') as input_file:
    sol_50_50=pickle.load(input_file)
with open('sol_60_40','rb') as input_file:
    sol_60_40=pickle.load(input_file)
with open('sol_70_30','rb') as input_file:
    sol_70_30=pickle.load(input_file)
