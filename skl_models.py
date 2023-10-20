# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:46:45 2023

@author: ichino

collection of models
"""
# example of models that can be used

listOfModels = []
listOfModelsNames = []

# from sklearn.linear_model import LinearRegression as mdl
# listOfModels.append(mdl())
# listOfModelsNames.append('linReg')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as mdl
for solverName in ['svd', 'lsqr', 'eigen']:
    listOfModels.append(mdl(solver = solverName))
    listOfModelsNames.append('lda_'+solverName)

from sklearn.neighbors import KNeighborsClassifier as mdl
for weightsParam in ['uniform', 'distance']:
    for n_neigh in [5,10,20,50]:#,100,200,500,1000,2000,5000]:#,10000,20000,50000]:
        listOfModels.append(mdl(n_neighbors = n_neigh, weights = weightsParam))
        listOfModelsNames.append('knn'+str(n_neigh)+weightsParam[0])

from sklearn.naive_bayes import GaussianNB as mdl
listOfModels.append(mdl())
listOfModelsNames.append('gNB')

from sklearn.tree import DecisionTreeClassifier as mdl
listOfModels.append(mdl())
listOfModelsNames.append('decTree')

# from sklearn.ensemble import RandomForestRegressor as mdl
# for maxDep in [5,10,20,50]:
#     listOfModels.append(mdl(max_depth = maxDep))
#     listOfModelsNames.append('rfReg'+str(maxDep))

from sklearn.ensemble import RandomForestClassifier as mdl
for maxDep in [5, 10, 20, 50]:
    for numEst in [5, 10, 20, 50, 70, 100]:
        listOfModels.append(mdl(n_estimators=numEst, max_depth=maxDep, ))
        listOfModelsNames.append('rfClass' + str(maxDep) + '-' + str(numEst))

from sklearn.neural_network import MLPClassifier as mdl
for layer_size in [5,10,20,54,108,[54,108,54],[10,100,10]]:
    listOfModels.append(mdl(hidden_layer_sizes=(layer_size), activation = 'logistic', max_iter=500))
    listOfModelsNames.append('nn'+str(layer_size))

list_of_models_and_models_names = [listOfModels,listOfModelsNames]