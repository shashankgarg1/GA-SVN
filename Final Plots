#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 22:40:21 2017

@author: shashank
"""

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np


lw=2
accuracies=[]
for i in range(len(sol_50_50)):
    accuracies.append(sol_50_50[i][3])
ind=np.argmax(accuracies)
y_test=sol_50_50[ind][9]
y_test = label_binarize(y_test, classes=['B','M'])
y_score=sol_50_50[ind][10]
precision_plot_50_50, recall_plot_50_50, _ = precision_recall_curve(y_test,y_score)
average_precision_plot_50_50 = average_precision_score(y_test, y_score)

accuracies=[]
for i in range(len(sol_60_40)):
    accuracies.append(sol_60_40[i][3])
ind=np.argmax(accuracies)
y_test=sol_60_40[ind][9]
y_test = label_binarize(y_test, classes=['B','M'])
y_score=sol_60_40[ind][10]
precision_plot_60_40, recall_plot_60_40, _ = precision_recall_curve(y_test,y_score)
average_precision_plot_60_40 = average_precision_score(y_test, y_score)

accuracies=[]
for i in range(len(sol_70_30)):
    accuracies.append(sol_70_30[i][3])
ind=np.argmax(accuracies)
y_test=sol_70_30[ind][9]
y_test = label_binarize(y_test, classes=['B','M'])
y_score=sol_70_30[ind][10]
precision_plot_70_30, recall_plot_70_30, _ = precision_recall_curve(y_test,y_score)
average_precision_plot_70_30 = average_precision_score(y_test, y_score)

plt.clf()
plt.plot(recall_plot_50_50, precision_plot_50_50, color='gold', lw=lw,
         label='Precision-recall curve for 50-50 Partition (area = {0:0.2f})'
               ''.format(average_precision_plot_50_50))
plt.plot(recall_plot_60_40, precision_plot_60_40, color='navy', lw=lw,
         label='Precision-recall curve for 60-40 Partition (area = {0:0.2f})'
               ''.format(average_precision_plot_60_40))
plt.plot(recall_plot_70_30, precision_plot_70_30, color='darkorange', lw=lw,
         label='Precision-recall curve for 70-30 Partition (area = {0:0.2f})'
               ''.format(average_precision_plot_70_30))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curves for various partitions')
plt.legend(loc="lower right")
plt.show()