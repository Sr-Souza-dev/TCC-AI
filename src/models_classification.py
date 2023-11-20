from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as ms
import numpy as np
import pickle

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


namesM = ['SVM', 'KNN', 'LR']
scoring = 'accuracy'

def GetModelsClassificationOptimized(dataName, trainSize):
    X = pd.read_csv(f'../Data/Cut/dataset1/X/Train_{trainSize}{dataName}.csv', sep=";")
    Y = pd.read_csv(f'../Data/Cut/dataset1/Y/Train_{trainSize}{dataName}.csv', sep=";")['OutPut_class |T+1|']

    print(f"             -- {dataName} - Inicio da otimização dos modelos de Classificação ")
    print(f'                {dataName} - X_train: {X.shape} | X_test: {X.shape} | Y_train: {Y.shape} | Y_test: {Y.shape}')

    # ----------------------------- Otimizando SVM ----------------------------------
    print(f'                 * {dataName} - SVM')
    SVM_grid = {
        'kernel': ['linear', 'poly'],
        'C': np.arange(0.01, 40, 15),
        'gamma': ['scale'],
        'degree': [1, 2, 3, 4],
        'coef0': np.arange(1, 4, 2),
    }
    svm_grid_search = GridSearchCV(estimator = SVC(), param_grid = SVM_grid, cv = 3, n_jobs = -1, verbose = 1, scoring = scoring)
    svm_grid_search.fit(X, Y)
    bestSVM = svm_grid_search.best_estimator_
    with open(f'../Results/optimization/classifications/SVM/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestSVM, f)
    
    
    # ----------------------------- Otimizando KNN ----------------------------------
    print(f'                 * {dataName} - KNN ')
    KNN_grid = {
        'n_neighbors': [3, 9, 17, 34],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'p': [1, 2]
    }
    knn_grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = KNN_grid, cv = 3, n_jobs = -1, verbose = 1, scoring = scoring)
    knn_grid_search.fit(X, Y)
    bestKNN = knn_grid_search.best_estimator_
    with open(f'../Results/optimization/classifications/KNN/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestKNN, f)

    # ----------------------------- Otimizando LR ----------------------------------
    print(f'                 * {dataName} - LR')
    LR_grid = {
        'penalty': ['l1', 'l2'],
        'solver': ['newton-cg', 'lbfgs'],
        'C': [10, 100, 1000],
        'max_iter': [50, 100, 500],
        'fit_intercept': [True, False],
        'class_weight': ['balanced', None],
        'warm_start': [True, False]
    }
    lr_grid_search = GridSearchCV(estimator = LogisticRegression(), param_grid = LR_grid, cv = 3, n_jobs = -1, verbose = 1, scoring = scoring)
    lr_grid_search.fit(X, Y)
    bestLR = lr_grid_search.best_estimator_
    with open(f'../Results/optimization/classifications/LR/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestLR, f)

    trainData = pd.concat([
        pd.Series(bestSVM.predict(X), name='SVM'),
        pd.Series(bestKNN.predict(X), name='KNN'),
        pd.Series(bestLR.predict(X), name='LR')
    ], axis=1)
    trainData.to_csv(f'../Results/train/classification/{dataName}_predictions.csv', sep=';', index=False)
    return bestSVM, bestKNN, bestLR


def GetModelsClassification(dataName):
    with open(f'../Results/optimization/classifications/SVM/{dataName}_model.pkl', 'rb') as f:
        SVM = pickle.load(f)
    with open(f'../Results/optimization/classifications/KNN/{dataName}_model.pkl', 'rb') as f:
        KNN = pickle.load(f)
    with open(f'../Results/optimization/classifications/LR/{dataName}_model.pkl', 'rb') as f:
        LR = pickle.load(f)

    return SVM, KNN, LR

def GetClassificationPredictions(dataName, Models, Names, X_test):
    results = pd.DataFrame()
    for model, name in zip(Models, Names):
        # print(type(model).__name__)
        series = pd.Series(name=name, data=model.predict(X_test.values))
        results = pd.concat([results, series], axis=1)


    results.to_csv(f'../Results/test/classification/{dataName}_predictions.csv', sep=';', index=False)
