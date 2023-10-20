
import pandas as pd
import numpy as np

import os
import sys
sys.path.insert(1, os.path.split(os.path.split(os.getcwd())[0])[0])
sys.path.insert(2, os.path.split(os.getcwd())[0])
import skLearn_wrapper as skw

# Create random data
np.random.seed(42)  # For reproducibility

exp = 3
num_rows = 10**exp  # You can change this to the desired number of rows

#%% dataframe creation
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

#%% dataset creation
ds = skw.dataset.Dataset(df, ['feature_{}'.format(i+1) for i in range(5)], 'label')

#%% example of dataset operations
print('\nreduced samples dataset')
print(ds.reduce_samples(12))
print(ds.reduce_samples(12).df)

bal_ds_day = ds.balance('day')
print('\nbalanced ds according to day')
print(bal_ds_day)
print(bal_ds_day.df)

bal_ds_day_group = ds.balance('day_group')
print('\nbalanced ds according to day_group')
print(bal_ds_day_group)
print(bal_ds_day_group.df)

print('classes are: {}'.format(ds.classes)) 
print('encoded labels as integers are: {}'.format(ds.y)) 
print('decoded labels as integers are: {}'.format(ds.y_decoded)) 
