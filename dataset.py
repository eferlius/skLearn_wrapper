import sklearn
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing

class Dataset:
    def __init__(self, pd_dataframe, features_list = [], class_col = ''):

        self.df = pd_dataframe # original dataframe
        if features_list == []:
            features_list = list(pd_dataframe.columns[:-1])
        self.features_list = features_list # list of strings: names of columns for features in the pd_dataframe
        
        if class_col == '':
            class_col = pd_dataframe.columns[-1]
        self.class_col = class_col # string: name of the column for labels in the pd_dataframe
        
        self.classes = list(set(self.df[class_col].values)) # name of the classes
        self.classes.sort()
        
        # encoding for the classes
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.classes)

        
        # extraction of the pandas df
        self.X = self.df[features_list].values # 2d array
        self.y_decoded = self.df[class_col].values # 1d array # can be strings
        self.y = self.encode_classes(self.df[class_col].values) # 1d array # are integers      
        

        self.n_samples = len(self.y)
        
    def encode_classes(self, classes):
        return self.le.transform(classes)
    
    def decode_classes(self, y):
        return self.le.inverse_transform(y)

    def __str__(self):
        return ('dataset of: \n- {n_samples} samples\n- {nfeat} features {featnames}\n- {nlabel} classes: {labnames}'.\
        format(n_samples = self.n_samples, nfeat = len(self.X[0]), featnames = str(self.features_list), 
               nlabel = len(np.unique(self.y)), labnames = str(self.classes)))
    
    def reduce_samples(self, n_samples = -1):
        if n_samples > 0:
            df = self.df.sample(n = n_samples)
            return Dataset(df.sort_index(), self.features_list, self.class_col)
        else:
            return self
        
    def balance(self, col_name):
        elements = self.df[col_name].values
            
        # count the number of occurrences of each class and gets the minimum
        min_counts = np.min(self.df[col_name].value_counts().values)
        
        # for every unique value
        it=-1
        for el in set(elements):
            it+=1
            el_df = self.df[self.df[col_name] == el]
            el_df_min = el_df.sample(n = min_counts)
            if it == 0:
                df_concat = el_df_min.copy()
            else:
                df_concat = pd.concat([df_concat, el_df_min], axis=0)
                
        return Dataset(df_concat.sort_index(), self.features_list, self.class_col)
        
    def get_test_train_samples(self, test_size = -1, n_samples = -1, 
                               group_col = '', test_group_col = '', balance_col = ''):
        
        assert test_size > -1 or (group_col != '' and test_group_col != ''),\
            "required test_size or group_col and test_group_col!"
            
        if balance_col != '':
            ds = self.balance(balance_col)
        else:
            ds = self
            
        if n_samples> -1 and n_samples < ds.n_samples:
            ds = ds.reduce_samples(n_samples)
        else:
            pass
        
        if group_col != '' and test_group_col != '':
            X_train = ds.df[ds.df[group_col] != test_group_col][ds.features_list].values
            X_test = ds.df[ds.df[group_col] == test_group_col][ds.features_list].values
            y_train = self.encode_classes(ds.df[ds.df[group_col] != test_group_col][ds.class_col].values)
            y_test = self.encode_classes(ds.df[ds.df[group_col] == test_group_col][ds.class_col].values)
        elif test_size > -1:
            if test_size == 0: # all goes to train
                X_train, X_test, y_train, y_test = ds.X, None, ds.y, None
            elif test_size == 1: # all goes to test
                X_train, X_test, y_train, y_test = None, ds.X, None, ds.y
            else: # calling the function of sklearn
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(ds.X, ds.y, test_size = test_size)
        
        return X_train, X_test, y_train, y_test

    

    # def scatterFeatures(self, n_samples = -1, featNames = None, labelColors = [''], mainTitle = '', alpha_scatter = 0.5, marker_scatter = '.', s_scatter = 1, alpha_hist = 0.5, bins = 20, superimpose = True, labelNames = None):
    #     '''
    #     Plots in a matrix one feature with respect to the other.
    #     On the diagonal, draws an histogram with the distribution of the given 
    #     feature for the different labels

    #     Parameters
    #     ----------
    #     n_samples : int, optional
    #         To reduce the number of samples.
    #         Please refer to reduceSamples(). 
    #         The default is -1.
    #     featNames : list of strings, optional
    #         features to be showed. The default is None, which means all the features
    #     Please refer to plots.scatterFeatures for the other parameters

    #     Returns
    #     -------
    #     Please refer to plots.scatterFeatures 
    #     '''
    #     # eventually reducing the samples. Keeping the tables to act on feat names
    #     _, _, X_table, y_table = self.reduceSamples(n_samples)

    #     # eventually considering only the given features
    #     if featNames == None:
    #         featNames = self.features_list
    #     if labelNames == None:
    #         labelNames = self.labels_names
    #     mainTitle = self.name + mainTitle


    #     # calling the function to plot
    #     fig, ax = plots.scatterFeatures(X_table[featNames].values, y_table, featNames, labelColors, mainTitle, alpha_scatter, marker_scatter, s_scatter, alpha_hist, bins, superimpose, labelNames)

    #     return fig, ax