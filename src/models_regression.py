from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as ms
import numpy as np
import pickle
import os

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


namesM = ['MLP', 'SVR', 'RF']
scoring = 'neg_mean_absolute_error'

def GetModelsRegressionOptimized(dataName, trainSize):
    X = pd.read_csv(f'../Data/Cut/dataset2/X/Train_{trainSize}{dataName}.csv', sep=";")
    Y = pd.read_csv(f'../Data/Cut/dataset2/Y/Train_{trainSize}{dataName}.csv', sep=";")['OutPut |T+1|']

    print(f"             -- {dataName} - Inicio da otimização dos modelos de Classificação ")
    print(f'                {dataName} - X_train: {X.shape} | X_test: {X.shape} | Y_train: {Y.shape} | Y_test: {Y.shape}')

    # ----------------------------- Otimizando MLP ----------------------------------
    print(f'                 * {dataName} - MLP')
    MLP_grid = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (30,80,50,20), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    mlp_grid_search = GridSearchCV(estimator = MLPRegressor(), param_grid = MLP_grid, cv = 3, n_jobs = -1, verbose = 1, scoring = scoring)
    mlp_grid_search.fit(X, Y)
    bestMLP = mlp_grid_search.best_estimator_
    dir_name = f'../Results/optimization/regression/MLPR'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(f'{dir_name}/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestMLP, f)
    
    
    # ----------------------------- Otimizando SVR ----------------------------------
    print(f'                 * {dataName} - SVR ')
    SVR_grid = {
        'kernel': ['linear', 'poly'],
        'C': np.arange(0.01, 40, 15),
        'gamma': ['scale'],
        'degree': [1, 2, 3],
        'coef0': np.arange(1, 4, 2),
    }
    SVR_grid_search = GridSearchCV(estimator = svm.SVR(), param_grid = SVR_grid, cv = 3, n_jobs = -1, verbose = 1, scoring = scoring)
    SVR_grid_search.fit(X, Y)
    bestSVR = SVR_grid_search.best_estimator_
    dir_name = f'../Results/optimization/regression/SVR'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(f'{dir_name}/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestSVR, f)

    # ----------------------------- Otimizando RF ----------------------------------
    print(f'                 * {dataName} - RF')
    RForest_grid = {
        'n_estimators': [100, 200, 300, 1000],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rforest_search = GridSearchCV(estimator = RandomForestRegressor(), param_grid = RForest_grid, cv = 3, n_jobs = -1, verbose = 1, scoring = scoring)
    rforest_search.fit(X, Y)
    bestRF = rforest_search.best_estimator_
    dir_name = f'../Results/optimization/regression/RF'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(f'{dir_name}/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestRF, f)

    trainData = pd.concat([
        pd.Series(bestMLP.predict(X), name='MLP'),
        pd.Series(bestSVR.predict(X), name='SVR'),
        pd.Series(bestRF.predict(X), name='RF')
    ], axis=1)
    trainData.to_csv(f'../Results/train/regression/{dataName}_predictions.csv', sep=';', index=False)
    return bestMLP, bestSVR, bestRF


def GetModelsRegression(dataName):
    with open(f'../Results/optimization/regression/MLPR/{dataName}_model.pkl', 'rb') as f:
        MLP = pickle.load(f)
    with open(f'../Results/optimization/regression/SVR/{dataName}_model.pkl', 'rb') as f:
        SVR = pickle.load(f)
    with open(f'../Results/optimization/regression/RF/{dataName}_model.pkl', 'rb') as f:
        RF = pickle.load(f)

    return MLP, SVR, RF


def GetRegressionPredictions(dataName, Names, Models, X_test, Y_test, X_train, Y_train):

    # Calcula o resultado no conjunto de treinameito
    results_ClassTrain = pd.DataFrame()
    for name, model in zip(Names, Models):
        series = pd.Series(name=name, data=(model.predict(X_train.values)).ravel())
        serie_last = series.shift(1)
        series_class = pd.Series(name=name, data = (series > serie_last).astype(int))
        results_ClassTrain = pd.concat([results_ClassTrain, series_class], axis=1)


    # Calcula o resultado no conjunto de teste
    results = pd.DataFrame()
    results_Class = pd.DataFrame()

    for name, model in zip(Names, Models):
        series = pd.Series(name=name, data=(model.predict(X_test.values)).ravel())
        serie_last = series.shift(1)
        series_class = pd.Series(name=name, data = (series > serie_last).astype(int))
        results_Class = pd.concat([results_Class, series_class], axis=1)
        results = pd.concat([results, series], axis=1)

    results_ClassTrain.to_csv(f'../Results/train/regression/{dataName}_predictions_class.csv', sep=';', index=False)
    results_Class.to_csv(f'../Results/test/regression/{dataName}_predictions_class.csv', sep=';', index=False)
    results.to_csv(f'../Results/test/regression/{dataName}_predictions.csv', sep=';', index=False)
