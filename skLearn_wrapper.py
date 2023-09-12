import numpy as np
from . import timer
from . import utils
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from pickle import dump, load
import os

# load(open(model_path),'rb')

def model_fit(model_to_train, X_train, y_train):
    return model_to_train.fit(X_train, y_train)

def model_predicts(trained_model, X_test):
    y_pred = trained_model.predict(X_test)
    y_pred_proba = trained_model.predict_proba(X_test)
    return y_pred, y_pred_proba

def compute_metrics(y_test, y_pred, round_res = 4):
    acc = np.around(accuracy_score(y_test, y_pred), round_res)
    cm = np.around(confusion_matrix(y_test, y_pred, normalize = 'true'), round_res)
    # use the two lines below to display the matrix
    # disp = ConfusionMatrixDisplay(cm, display_labels = np.unique(y_train))
    # disp.plot()
    prec, recall, fscore, support = np.around(precision_recall_fscore_support(y_test, y_pred), round_res)
    # spec = prec
    # sens = recall
    return acc, prec, recall, fscore, support, cm

def analyze_trained_model(trained_model, X_test, y_test):
    # make prediction
    y_pred, y_pred_proba = model_predicts(trained_model, X_test)
    # compute metrics
    acc, prec, recall, fscore, support, cm = compute_metrics(y_test, y_pred)
    return y_pred, y_pred_proba, acc, prec, recall, fscore, support, cm
    
def analyze_model_from_ds(ds_dataset_train_test, model_to_train, test_size = -1, n_samples = -1, 
                           group_col = '', test_group_col = '', balance_col = '',
                           round_res = 4):
    # get train and test samples from dataset
    X_train, X_test, y_train, y_test = ds_dataset_train_test.get_test_train_samples(test_size = test_size, 
                                       n_samples = n_samples, group_col = group_col, 
                                       test_group_col = test_group_col, balance_col = balance_col)
    
    # train the model
    trained_model = model_fit(model_to_train, X_train, y_train)
    # analyze the trained model
    y_pred, y_pred_proba, acc, prec, recall, fscore, support, cm = analyze_trained_model(trained_model, X_test, y_test)
    
    return trained_model, y_pred, y_pred_proba, acc, prec, recall, fscore, support, cm

    
def test_models_from_ds(ds_dataset_train_test, list_of_models_and_models_names, 
                test_size = -1, group_col = '', balance_col = '',
                start_n_samples = 0, multiplier = 10, n_runs_per_model = 5, stop_n_samples = 0,
                max_time_per_run_seconds = 300, output_file = '', round_res = 4,
                models_saving_path = '', verbose = 'only models'):
    
    assert test_size > -1 or group_col != '',\
    "required test_size or group_col"
    
    assert verbose in [True, False, 'only models'], \
        "Input must be True, False, or 'only models'"
    
    if start_n_samples == 0 or start_n_samples > ds_dataset_train_test.n_samples:
        start_n_samples = ds_dataset_train_test.n_samples
    if stop_n_samples == 0 or stop_n_samples > ds_dataset_train_test.n_samples:
        stop_n_samples = ds_dataset_train_test.n_samples
        
    if output_file != '':
        newRow = ['code', 'modelName', 'test_group_col', 'acc', 'n_samples', 'elabTime', 'trainTime', 'testTime']
        utils.write_row_csv(output_file, newRow, mode = 'a')
    listOfModels,listOfModelsNames = list_of_models_and_models_names
    for model, modelName in zip(listOfModels,listOfModelsNames):
        time_exceeded = False
        samples_exceeded = False
        n_samples = start_n_samples
        elabTime = 0 # time to train and test the algorithm
        if verbose == 'only models':
            print('testing {}'.format(modelName))
        while n_samples <= stop_n_samples and n_samples >= -1 and elabTime <= max_time_per_run_seconds and not samples_exceeded:
            try: # to avoid knn crashing if n_samples < n_neighbours
                if group_col != '':
                    test_group_cols = list(set(ds_dataset_train_test.df[group_col]))
                    test_group_cols.sort()
                else:
                    test_group_cols = ' ' # empty value, as the default of get_test_train_samples
                for test_group_col in test_group_cols:
                    for run in range(n_runs_per_model):
                        # split
                        X_train, X_test, y_train, y_test = ds_dataset_train_test.get_test_train_samples(test_size = test_size, 
                                               group_col = group_col, test_group_col = test_group_col , 
                                               n_samples = n_samples, balance_col = balance_col)
                        
                        code = utils.this_moment(fmt = '%Y%m%d %H%M%S.%f')
                        t0 = timer.Timer()
                        
                        # train
                        model = model_fit(model, X_train, y_train)
                        trainTime = np.around(t0.lap(printTime = False), round_res)
    
                        # test
                        y_pred, y_pred_proba = model_predicts(model, X_test)
                        testTime = np.around(t0.lap(printTime = False), round_res)
                        
                        # metrics
                        acc = np.around(accuracy_score(y_test, y_pred), round_res)
                        elabTime = np.around(t0.stop(printTime = False),round_res)
                        
                        if verbose == True:
                            # log
                            toPrint = '{} - {} ({}): {:.2f}% - test on {}: {} with {}'.format\
                            (code, modelName, run, acc*100, 
                             (group_col if test_group_col != ' ' else ''),
                             (test_group_col if test_group_col != ' ' else 'all'), n_samples)
                            print(toPrint)
                        
                        if output_file != '':
                            newRow = [code, modelName, str(group_col)+str(test_group_col), acc, n_samples, elabTime, trainTime, testTime]
                            utils.write_row_csv(output_file, newRow, mode = 'a')
    
                        if elabTime > max_time_per_run_seconds:
                            time_exceeded = True
                            
                        if models_saving_path != '':
                            # save the model
                            dump(model, open(os.path.join(models_saving_path, '{} {} {} {}.pkl'.\
                                                          format(code, str(group_col), str(test_group_col),modelName)), 'wb'))
                            
                        if time_exceeded: # out of the run
                            break
                    if time_exceeded: # out of the test column
                        break
                if time_exceeded: # out of the n samples
                    if verbose == True:
                        toPrint = '{modelName} w/ {n_samples} samples took {elabTime:0.2f}s while max is {maxTime}s -> next model'\
                                .format(modelName = modelName, n_samples = n_samples, elabTime = elabTime, maxTime = max_time_per_run_seconds)
                        print(toPrint)
                    break
            except:
                if verbose == True:
                    toPrint = '{modelName} w/ {n_samples} samples throwing exception -> increase num samples'.format(modelName = modelName, n_samples = n_samples)
                    print(toPrint)
            
            finally:
                if n_samples == stop_n_samples:
                    samples_exceeded = True
                # prepare for next iter
                n_samples *= multiplier
                if n_samples > stop_n_samples:
                    n_samples = stop_n_samples
                if n_samples > ds_dataset_train_test.n_samples:
                    n_samples = ds_dataset_train_test.n_samples
                

                   