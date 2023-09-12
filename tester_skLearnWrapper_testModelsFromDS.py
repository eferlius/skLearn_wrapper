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

cwd = os.getcwd()

output_file = os.path.join(cwd,'here.csv')
output_folder = os.path.join(cwd,'models')

# Create random data
np.random.seed(42)  # For reproducibility

t = timer.Timer()
exp = 4
num_rows = 5*10**exp  # You can change this to the desired number of rows
print(num_rows)
features = {
    'feature_1': np.random.rand(num_rows),
    'feature_2': np.random.rand(num_rows),
    'feature_3': np.random.rand(num_rows),
    'feature_4': np.random.rand(num_rows),
    'feature_5': np.random.rand(num_rows)
}
day_column = np.random.choice(['d{}'.format(i) for i in range(5)], num_rows).tolist()
group_column = np.random.choice(['g{}'.format(i) for i in range(int(8))], num_rows)
label_column = np.random.choice(['a','b','c','d'], num_rows)

# Create DataFrame
data = {'day': day_column, **features, 'group': group_column, 'label': label_column}
df = pd.DataFrame(data)
# concatenate two columns to get unique values
df['day_group']=df['day'].values+'_'+df['group'].values

print(df)

listOfModels = []
listOfModelsNames = []
# from sklearn.linear_model import LinearRegression as mdl
# listOfModels.append(mdl())
# listOfModelsNames.append('linReg')

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
ds = dataset.Dataset(df, ['feature_{}'.format(i+1) for i in range(5)], 'label')

lda_svd = skLearn_wrapper.model_fit(mdl(solver = 'svd'), ds.X, ds.y)

trained_model, y_pred, y_pred_proba, acc, prec, recall, fscore, support, cm = \
    skLearn_wrapper.analyze_model_from_ds(ds, lda_svd, test_size = 0.3)
    
disp = ConfusionMatrixDisplay(cm, display_labels = ds.classes)
disp.plot()
    
# import matplotlib.pyplot as plt
# plt.plot(disp)

skLearn_wrapper.test_models_from_ds(ds, list_of_models_and_models_names, group_col  = 'day', test_size = 0.2,
                            n_runs_per_model = 2, start_n_samples = 1000, multiplier = 5, stop_n_samples = 30000, 
                            balance_col = '', output_file = output_file,
                            models_saving_path = '')
print('done')