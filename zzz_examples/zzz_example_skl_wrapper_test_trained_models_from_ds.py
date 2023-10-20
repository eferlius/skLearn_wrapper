import pandas as pd
import numpy as np
import os

import sys
sys.path.insert(1, os.path.split(os.path.split(os.getcwd())[0])[0])
sys.path.insert(2, os.path.split(os.getcwd())[0])
import skLearn_wrapper as skw

cwd = os.getcwd()
output_file = os.path.join(cwd,'here.csv')
# Create random data
np.random.seed(42)  # For reproducibility
#%% train dataframe creation
exp = 4
num_rows = 10**exp
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

#%% train dataset creation
ds = skw.dataset.Dataset(df, ['feature_{}'.format(i+1) for i in range(5)], 'label')

#%% model training
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as mdl
lda_svd = skw.skl_wrapper.model_fit(mdl(solver = 'svd'), ds.X, ds.y)

#%% train dataframe creation
num_rows = int(num_rows/10)
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
df_test = pd.DataFrame(data)
# concatenate two columns to get unique values
df_test['day_group']=df_test['day'].values+'_'+df_test['group'].values
print(df_test)

#%% test dataset creation
ds_test = skw.dataset.Dataset(df_test, ['feature_{}'.format(i+1) for i in range(5)], 'label')

#%% test the model
listOfTrainedModels = [lda_svd]
listOfModelsNames = ['lda_svd']
list_of_models_and_models_names = [listOfTrainedModels,listOfModelsNames]

skw.skl_wrapper.test_trained_models_from_ds(ds_test, list_of_models_and_models_names, group_col  = 'day', test_size = 0.2,
                            n_runs_per_model = 2, start_n_samples = 1000, multiplier = 5, stop_n_samples = 30000, 
                            balance_col = '', output_file = output_file)
print('done')