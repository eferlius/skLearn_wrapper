import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from pickle import dump, load
import os

from . import _timer_
from . import _utils_

# dump(model, open(model_path, 'wb'))
# load(open(model_path,'rb'))

def model_fit(model_to_train, X_train, y_train):
    '''
    Trains a model with X_train and y_train data.

    Parameters
    ----------
    model_to_train : sklearn model
        Model to train.
    X_train : matrix n_samples*n_features
        Features for training.
    y_train : array n_samples
        Classes for training.

    Returns
    -------
    trained_model : sklearn model
        Trained model

    '''
    return model_to_train.fit(X_train, y_train)

def model_predict(trained_model, X_test):
    '''
    Run trained model on X_test data.

    Parameters
    ----------
    trained_model : sklearn model
        Already trained.
    X_test : matrix n_samples*n_features
        Data for prediction.

    Returns
    -------
    y_pred : array n_samples
        Predicted classes.
    y_pred_proba : matrix n_samples*n_classes
        Probability of each class for each sample.

    '''
    y_pred = trained_model.predict(X_test)
    y_pred_proba = trained_model.predict_proba(X_test)
    return y_pred, y_pred_proba

def compute_metrics(y_test, y_pred, round_res = 4, normalize_cm = 'true'):
    '''
    Given ground truth and predicted classes, computes metrics to evaluate the 
    classifier performances.

    Parameters
    ----------
    y_test : array n_samples
        Ground truth classes.
    y_pred : array n_samples
        Predicted classes.
    round_res : int, optional
        Number of digits after comma for results rounding. The default is 4.
    normalize_cm : string, optional
        How to normalize the computations of the confusion matrix. The default is 'true'.

    Returns
    -------
    accuracy : float
        accuracy (tp + tn) / (tp + fp + fn + tn).
    precision : array of float
        precision tp / (tp + fp)
        The precision is intuitively the ability of the classifier not to label a negative sample as positive.
    recall : array of float
        recall (or sensitivity) tp / (tp + fn)
        The recall is intuitively the ability of the classifier to find all the positive samples.
    fscore : array of float
        fscore 2 * (precision * recall) / (precision + recall).
    support : array of int
        support number of items of each class in the y_test.
    cm : matrix n_class*n_class
        Confusion matrix.
    '''
    accuracy = np.around(accuracy_score(y_test, y_pred), round_res)
    cm = np.around(confusion_matrix(y_test, y_pred, normalize = normalize_cm ), round_res)
    # use the two lines below to display the matrix
    # disp = ConfusionMatrixDisplay(cm, display_labels = np.unique(y_train))
    # disp.plot()
    precision, recall, fscore, support = np.around(precision_recall_fscore_support(y_test, y_pred), round_res)
    # spec = prec
    # sens = recall
    return accuracy, precision, recall, fscore, support, cm

def analyze_trained_model(trained_model, X_test, y_test, round_res = 4, normalize_cm = 'true'):
    '''
    Given a trained model, X_test features for prediction and y_test as ground truth,
    first executes classification and then evaluates the performances.

    Parameters
    ----------
    trained_model : sklearn model
        Already trained
    X_test : matrix n_samples*n_features
        Features for prediction
    y_test : array n_samples
        Ground truth classes
    round_res : int, optional
        Number of digits after comma for results rounding. The default is 4.
    normalize_cm : string, optional
        How to normalize the computations of the confusion matrix. The default is 'true'
        
    Returns
    -------
    y_pred : array n_samples
        predicted classes.
    y_pred_proba : matrix n_samples*n_classes
        Probability of each class for each sample.
    accuracy : float
        accuracy (tp + tn) / (tp + fp + fn + tn).
    precision : array of float
        precision tp / (tp + fp)
        The precision is intuitively the ability of the classifier not to label a negative sample as positive.
    recall : array of float
        recall (or sensitivity) tp / (tp + fn)
        The recall is intuitively the ability of the classifier to find all the positive samples.
    fscore : array of float
        fscore 2 * (precision * recall) / (precision + recall).
    support : array of int
        support number of items of each class in the y_test.
    cm : matrix n_class*n_class
        Confusion matrix.

    '''
    # make prediction
    y_pred, y_pred_proba = model_predict(trained_model, X_test)
    # compute metrics
    accuracy, precision, recall, fscore, support, cm = compute_metrics(y_test, y_pred, round_res, normalize_cm)
    return y_pred, y_pred_proba, accuracy, precision, recall, fscore, support, cm
    
def analyze_model_from_ds(ds_dataset_train_test, model_to_train, test_size = -1, n_samples = -1, 
                           group_col = '', test_group_col = '', balance_col = '',
                           round_res = 4, normalize_cm = 'true'):
    '''
    Given a dataset and a model to be trained, first trains it according to the 
    inputs and presents the metrics.

    Parameters
    ----------
    ds_dataset_train_test : Dataset object
        Dataset containing features, classes and eventual other columns for grouping
    model_to_train : sklearn model
        Model to train
    test_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
        If int, represents the absolute number of test samples.
        If 1, all the samples goes to test
        If 0, all the samples goes to train
        If -1, the selection is made according to test_group_col
        The default is -1.
    n_samples : int, optional
        Number of samples to be considered
        If -1, the number of samples is not modified    
        The default is -1.
    group_col : string, optional
        Name of the column in the original dataframe to be used for grouping in 
        test and training. 
        The default is '', which means that no grouping will be executed.
    test_group_col : string, optional
        All the rows whose value in the column "group_col" corresponding to this 
        value will be used for testing. The others for training.
        The default is '', which means that no grouping will be executed.
    balance_col : string, optional
        Name of the column in the original dataframe to be used for balancing the dataset. 
        The default is '', which means that no balancing will be executed.
    round_res : int, optional
        Number of digits after comma for results rounding. The default is 4.
    normalize_cm : string, optional
        How to normalize the computations of the confusion matrix. The default is 'true'

    Returns
    -------
    trained_model : sklearn model
        Trained model.
    y_pred : array n_samples
        predicted classes.
    y_pred_proba : matrix n_samples*n_classes
        Probability of each class for each sample.
    accuracy : float
        accuracy (tp + tn) / (tp + fp + fn + tn).
    precision : array of float
        precision tp / (tp + fp)
        The precision is intuitively the ability of the classifier not to label a negative sample as positive.
    recall : array of float
        recall (or sensitivity) tp / (tp + fn)
        The recall is intuitively the ability of the classifier to find all the positive samples.
    fscore : array of float
        fscore 2 * (precision * recall) / (precision + recall).
    support : array of int
        support number of items of each class in the y_test.
    cm : matrix n_class*n_class
        Confusion matrix.

    '''
    # get train and test samples from dataset
    X_train, X_test, y_train, y_test = ds_dataset_train_test.get_test_train_samples(test_size = test_size, 
                                       n_samples = n_samples, group_col = group_col, 
                                       test_group_col = test_group_col, balance_col = balance_col)
    
    # train the model
    trained_model = model_fit(model_to_train, X_train, y_train)
    # analyze the trained model
    y_pred, y_pred_proba, accuracy, precision, recall, fscore, support, cm = analyze_trained_model(trained_model, X_test, y_test, round_res, normalize_cm)
    
    return trained_model, y_pred, y_pred_proba, accuracy, precision, recall, fscore, support, cm

    
def test_models_from_ds(ds_dataset_train_test, list_of_models_and_models_names, 
                test_size = -1, group_col = '', balance_col = '',
                start_n_samples = 0, multiplier = 10, stop_n_samples = 0, n_runs_per_model = 5,
                max_time_per_run_seconds = 300, output_file = '', round_res = 4,
                models_saving_path = '', verbose = 'only models'):
    '''
    Given a dataset and a list of models to be trained, iteratively trains them 
    with an increasing number of samples. 
    Passes to the next model if max_time_per_run_seconds is exceeded.
    Writes training and test time and accuracy on output_file, if specified.
    Saves the model in models_saving_path with a univoque code with moment and 
    model name, if specified.
    
    Parameters
    ----------
    ds_dataset_train_test : Dataset object
        Dataset containing features, classes and eventual other columns for grouping.
    list_of_models_and_models_names : list of 2 lists
        First list contains the models to be trained, the second list the names of the models.
    test_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
        If int, represents the absolute number of test samples.
        If 1, all the samples goes to test.
        If 0, all the samples goes to train.
        If -1, the selection is made according to test_group_col.
        The default is -1.
    n_samples : int, optional
        Number of samples to be considered.
        The default is -1, which means that the number of samples is not modified.
    group_col : string, optional
        Name of the column in the original dataframe to be used for grouping in test and training. 
        The default is '', which means that no grouping will be executed.
    balance_col : string, optional
        Name of the column in the original dataframe to be used for balancing the dataset. 
        The default is '', which means that no balancing will be executed.
    start_n_samples : int, optional
        Number of samples in the first iteration
        The default is 0, which means that the start samples will be the total amount of samples.
    multiplier : int, optional
         Factor to increase the number of samples at each iteration. The default is 10.
    stop_n_samples : int, optional
        Max number of samples when testing the model.
        The default is 0, which means that the stop samples will be the total amount of samples.
    n_runs_per_model : int, optional
        Number of runs for each model with each number of samples. The default is 5.
    max_time_per_run_seconds : float, optional
        Maximum time elapsed to train and test a model for each iteration. 
        If exceeded, the algorithm passes to the next model. The default is 300.
    output_file : string, optional
        Path to a csv file to write:
            ['code', 'modelName', 'test_group_col', 'acc', 'n_samples', 'elabTime', 'trainTime', 'testTime']
        The default is '', which implies that no output_file is written.
    round_res : int, optional
        Number of digits after comma for results rounding. The default is 4.
    models_saving_path : string, optional
        Path to a folder where to save the trained models. 
        The default is '', which implies that models are not saved.
    verbose : string or boolean, optional
        What to print during the execution of the loop. The default is 'only models'.

    Returns
    -------
    None.

    '''
    assert test_size > -1 or group_col != '',\
    "required test_size or group_col"
    
    assert verbose in [True, False, 'only models'], \
        "verbose must be True, False, or 'only models'"
    
    if start_n_samples == 0 or start_n_samples > ds_dataset_train_test.n_samples:
        start_n_samples = ds_dataset_train_test.n_samples
    if stop_n_samples == 0 or stop_n_samples > ds_dataset_train_test.n_samples:
        stop_n_samples = ds_dataset_train_test.n_samples
        
    if output_file != '':
        newRow = ['code', 'modelName', 'test_group_col', 'acc', 'n_samples', 'elabTime', 'trainTime', 'testTime']
        _utils_.write_row_csv(output_file, newRow, mode = 'a')
    # unpack list    
    listOfModels, listOfModelsNames = list_of_models_and_models_names
    
    for model, modelName in zip(listOfModels,listOfModelsNames):
        time_exceeded = False
        samples_exceeded = False
        n_samples = start_n_samples
        elabTime = 0 # time to train and test the algorithm
        if verbose == 'only models':
            print('testing {}'.format(modelName))
        while n_samples <= stop_n_samples and n_samples >= -1 and elabTime <= max_time_per_run_seconds and not samples_exceeded:
            try: 
                # if grouping column
                if group_col != '':
                    # get list of all possible values
                    test_group_cols = list(set(ds_dataset_train_test.df[group_col]))
                    test_group_cols.sort()
                else:
                    test_group_cols = ' ' # empty value, as the default of get_test_train_samples
                for test_group_col in test_group_cols:
                    for run in range(n_runs_per_model):
                        # split (eventually according to group col)
                        X_train, X_test, y_train, y_test = ds_dataset_train_test.get_test_train_samples(test_size = test_size, 
                                               group_col = group_col, test_group_col = test_group_col, 
                                               n_samples = n_samples, balance_col = balance_col)
                        # univoque code for test
                        code = _utils_.this_moment(fmt = '%Y%m%d %H%M%S.%f')
                        t0 = _timer_.Timer()
                        
                        # train
                        model = model_fit(model, X_train, y_train)
                        trainTime = np.around(t0.lap(printTime = False), round_res)
    
                        # test
                        y_pred, y_pred_proba = model_predict(model, X_test)
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
                            _utils_.write_row_csv(output_file, newRow, mode = 'a')
    
                        if elabTime > max_time_per_run_seconds:
                            time_exceeded = True
                            
                        if models_saving_path != '':
                            os.makedirs(models_saving_path, exist_ok = True)
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
                # prepare for next iteration
                n_samples *= multiplier
                if n_samples > stop_n_samples:
                    n_samples = stop_n_samples
                if n_samples > ds_dataset_train_test.n_samples:
                    n_samples = ds_dataset_train_test.n_samples
                

                   