# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:27:50 2023

@author: ichino
"""

import pandas as pd
import numpy as np
import dataset
import timer
import skLearn_wrapper
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.datasets import load_iris

cwd = os.getcwd()
output_file = os.path.join(cwd,'here.csv')
output_folder = os.path.join(cwd,'models')

df = load_iris(as_frame = True).frame

listOfModels = []
listOfModelsNames = []

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as mdl
for solverName in ['svd', 'lsqr', 'eigen']:
    listOfModels.append(mdl(solver = solverName))
    listOfModelsNames.append('lda_'+solverName)

# from sklearn.neighbors import KNeighborsClassifier as mdl
# for weightsParam in ['uniform', 'distance']:
#     for n_neigh in [5,10,20,50]:#,100,200,500,1000,2000,5000]:#,10000,20000,50000]:
#         listOfModels.append(mdl(n_neighbors = n_neigh, weights = weightsParam))
#         listOfModelsNames.append('knn'+str(n_neigh)+weightsParam[0])
        
list_of_models_and_models_names = [listOfModels,listOfModelsNames]

# prepare daset
ds = dataset.Dataset(df, list(df.columns[:-1]), df.columns[-1])
ds = dataset.Dataset(df.sample(frac = 1))

test_size = 20

lda_svd = skLearn_wrapper.model_fit(mdl(solver = 'svd'), ds.X[:-test_size], ds.y[:-test_size])

y_pred, y_pred_proba, acc, prec, recall, fscore, support, cm = \
    skLearn_wrapper.analyze_trained_model(lda_svd, ds.X[-test_size:], ds.y[-test_size:])
    
print(acc)
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels = ds.classes)
disp.plot()

trained_model, y_pred, y_pred_proba, acc, prec, recall, fscore, support, cm = \
    skLearn_wrapper.analyze_model_from_ds(ds, lda_svd, test_size = test_size)
    
disp = ConfusionMatrixDisplay(cm, display_labels = ds.classes)
disp.plot()
    

skLearn_wrapper.test_models_from_ds(ds, list_of_models_and_models_names, group_col  = '', test_size = 0.2,
                            n_runs_per_model = 2, start_n_samples = 10, multiplier = 5, stop_n_samples = 30000, 
                            balance_col = '', output_file = output_file,
                            models_saving_path = output_folder)
