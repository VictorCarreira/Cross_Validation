#-*- coding: utf-8 -*-
#This program is an implementation of a Cross-Validation test
#Author: Victor Carreira

###################################### THEORY ################################################
#   Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally. Here is a flowchart of typical cross validation workflow in model training. The best parameters can be determined by grid search techniques.
#   Cross-validation, sometimes called rotation estimation[1][2][3] or out-of-sample testing, is any of various similar model validation techniques for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice. In a prediction problem, a model is usually given a dataset of known data on which training is run (training dataset), and a dataset of unknown data (or first seen data) against which the model is tested (called the validation dataset or testing set).[4][5] The goal of cross-validation is to test the model's ability to predict new data that was not used in estimating it, in order to flag problems like overfitting or selection bias[6] and to give an insight on how the model will generalize to an independent dataset (i.e., an unknown dataset, for instance from a real problem).

# - References:
#   [4] Galkin, Alexander (November 28, 2011). "What is the difference between test set and validation set?". Retrieved 10 October 2018.
#   [5] Newbie question: Confused about train, validation and test data!". Archived from the original on 2015-03-14. Retrieved 2013-11-14.
##############################################################################################


################################ REQUIRED PACKAGES  #######################################
import numpy as np
import pylab as py
import scipy as sp  
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
###########################################################################################
