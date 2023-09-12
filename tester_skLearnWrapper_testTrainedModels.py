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


#%% training dataframe
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

ds = dataset.Dataset(df, ['feature_{}'.format(i+1) for i in range(5)], 'label')
#%% train the model
# train the model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as mdl
lda_svd = skLearn_wrapper.model_fit(mdl(solver = 'svd'), ds.X, ds.y)

#%% test dataframe
num_rows = int(num_rows/10)
features = {
    'feature_1': np.random.rand(num_rows),
    'feature_2': np.random.rand(num_rows),
    'feature_3': np.random.rand(num_rows),
    'feature_4': np.random.rand(num_rows),
    'feature_5': np.random.rand(num_rows)
}
day_column = np.random.choice(['day{}'.format(i) for i in range(5)], num_rows).tolist()
group_column = np.random.choice(['group{}'.format(i) for i in range(int(8))], num_rows)
label_column = np.random.choice(['a','b','c','d'], num_rows)

# Create DataFrame
data = {'day': day_column, **features, 'group': group_column, 'label': label_column}
df_test = pd.DataFrame(data)
# concatenate two columns to get unique values
df_test['day_group']=df_test['day'].values+'_'+df_test['group'].values

print(df_test)

ds_test = dataset.Dataset(df_test, ['feature_{}'.format(i+1) for i in range(5)], 'label')

#%% test the model
listOfTrainedModels = [lda_svd]
listOfModelsNames = ['lda_svd']
list_of_models_and_models_names = [listOfTrainedModels,listOfModelsNames]

skLearn_wrapper.test_trained_models_from_ds(ds_test, list_of_models_and_models_names, group_col  = 'day', test_size = 0.2,
                            n_runs_per_model = 2, start_n_samples = 1000, multiplier = 5, stop_n_samples = 30000, 
                            balance_col = '', output_file = output_file)
print('done')