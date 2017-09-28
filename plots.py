#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 22:37:50 2017

@author: shashank
"""

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

def plot_graph(sol):
    accuracies=[]
    for i in range(len(sol)):
        accuracies.append(sol[i][3])
    ind=np.argmax(accuracies)
    y_test=sol[ind][9]
    y_test = label_binarize(y_test, classes=['B','M'])
    y_score=sol[ind][10]
    precision_plot, recall_plot, _ = precision_recall_curve(y_test,y_score)
    average_precision_plot = average_precision_score(y_test, y_score)

    # Compute micro-average ROC curve and ROC area
    precision_micro, recall_micro, _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
    average_precision_micro = average_precision_score(y_test, y_score,average="micro")
    lw = 2

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall_plot, precision_plot, lw=lw, color='navy',label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision_plot))
    plt.legend(loc="lower left")
    plt.show()
    
    ###############################################################################
    
    plt.clf()
    plt.plot(recall_micro, precision_micro, color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision_micro))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('micro-average Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

    #plotting the worst output
    ind=np.argmin(accuracies)
    y_test=sol[ind][9]
    y_test = label_binarize(y_test, classes=['B','M'])
    y_score=sol[ind][10]
    precision_plot, recall_plot, _ = precision_recall_curve(y_test,y_score)
    average_precision_plot = average_precision_score(y_test, y_score)

    # Compute micro-average ROC curve and ROC area
    precision_micro, recall_micro, _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
    average_precision_micro = average_precision_score(y_test, y_score,average="micro")
    lw = 2
    
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall_plot, precision_plot, lw=lw, color='navy',
         label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision_plot))
    plt.legend(loc="lower left")
    plt.show()

    ###############################################################################

    #plotting the average output
    ind=1
    y_test=sol[ind][9]
    y_test = label_binarize(y_test, classes=['B','M'])
    y_score=sol[ind][10]
    precision_plot, recall_plot, _ = precision_recall_curve(y_test,
                                                            y_score)
    average_precision_plot = average_precision_score(y_test, y_score)
    
    # Compute micro-average ROC curve and ROC area
    precision_micro, recall_micro, _ = precision_recall_curve(y_test.ravel(),
                                                              y_score.ravel())
    average_precision_micro = average_precision_score(y_test, y_score,
                                                     average="micro")
    lw = 2

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall_plot, precision_plot, lw=lw, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision_plot))
    plt.legend(loc="lower left")
    plt.show()
    
    ###############################################################################
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))