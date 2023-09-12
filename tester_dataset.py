# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 19:04:54 2023

@author: ichino
"""

import pandas as pd
import numpy as np
import dataset
import timer

# Create random data
np.random.seed(42)  # For reproducibility

t = timer.Timer()
for exp in range(2,4):
    num_rows = 5*10**exp  # You can change this to the desired number of rows
    print(num_rows)
    features = {
        'feature_1': np.random.rand(num_rows),
        'feature_2': np.random.rand(num_rows),
        'feature_3': np.random.rand(num_rows),
        'feature_4': np.random.rand(num_rows),
        'feature_5': np.random.rand(num_rows)
    }
    day_column = np.random.choice(['d{}'.format(i) for i in range(4)], num_rows).tolist()
    group_column = np.random.choice(['g{}'.format(i) for i in range(int(num_rows/4))], num_rows)
    label_column = np.random.choice(['l{}'.format(i) for i in range(int(4))], num_rows)
    
    # Create DataFrame
    data = {'day': day_column, **features, 'group': group_column, 'label': label_column}
    df = pd.DataFrame(data)
    t.elap('df pandas')
    # concatenate two columns to get unique values
    df['day_group']=df['day'].values+'_'+df['group'].values
    t.elap('concatenation')
    
    ds = dataset.Dataset(df, ['feature_{}'.format(i+1) for i in range(5)], 'label')
    t.elap('dataset')
    # print(ds.reduce_samples(12).df)

    bal_ds = ds.balance('day')
    t.elap('balance day')
    bal_ds = ds.balance('day_group')
    t.elap('balance day_group')
    
    ds.get_test_train_samples(group_col = 'day', test_group_col = 'd1')
    t.elap('get_test_train_samples')
    
    print(ds.classes)
    
    print(ds.y)
    print(ds.y_decoded)
    
    print(ds.encode_classes(['l0', 'l3']))
    print(ds.decode_classes([1,2,3]))
    # bal_ds = ds.balance2('day')
    # t.elap('balance2 day')
    # bal_ds = ds.balance2('day_group')
    # t.elap('balance2 day_group')
    
    # red_ds = ds.reduce_samples(int(num_rows/2))
    # t.elap('reduce samples')
    
    # X, y = ds.get_reduced_samples(int(num_rows/2))
    # t.elap('get reduced samples')
    # X, y = ds.get_reduced_samples2(int(num_rows/2))
    # t.elap('get reduced samples2')
    
    
    # print(bal_ds.df)
    # print(bal_ds2.df)
