# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:27:50 2023

@author: ichino
"""
import pandas as pd
import numpy as np

import os
import sys
sys.path.insert(1, os.path.split(os.path.split(os.getcwd())[0])[0])
import skLearn_wrapper as skw

cwd = os.getcwd()
output_file = os.path.join(cwd,'here.csv')
output_folder = os.path.join(cwd,'models')
# Create random data
np.random.seed(42)  # For reproducibility

#%% initialization of models to be tested
listOfModels = []
listOfModelsNames = []

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as mdl
for solverName in ['svd', 'lsqr', 'eigen']:
    listOfModels.append(mdl(solver = solverName))
    listOfModelsNames.append('lda_'+solverName)
list_of_models_and_models_names = [listOfModels,listOfModelsNames]
#%% dataframe creation
exp = 4
num_rows = 10**exp
np.random.seed(42)  # For reproducibility
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

# create DataFrame
data = {'day': day_column, **features, 'group': group_column, 'label': label_column}
df = pd.DataFrame(data)
# concatenate two columns to get unique values
df['day_group']=df['day'].values+'_'+df['group'].values
print('original dataframe')
print(df)

#%% dataset creation
ds = skw.dataset.Dataset(df, ['feature_{}'.format(i+1) for i in range(5)], 'label')
#%% model training
cwd = os.getcwd()
output_file = os.path.join(cwd,'here.csv')
output_folder = os.path.join(cwd,'models')
skw.skl_wrapper.test_models_from_ds(ds, list_of_models_and_models_names, group_col  = 'day', test_size = 0.2,
                            n_runs_per_model = 2, start_n_samples = 1000, multiplier = 5, stop_n_samples = 30000, 
                            balance_col = '', output_file = output_file,
                            models_saving_path = output_folder)
print('done')