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

namesM = ['SVM', 'KNN', 'LR']
scoring = 'accuracy'

def GetModelsClassificationOptimized(dataName, trainSize):
    X = pd.read_csv(f'../Data/Cut/dataset1/X/Train_{trainSize}{dataName}.csv', sep=";")
    Y = pd.read_csv(f'../Data/Cut/dataset1/Y/Train_{trainSize}{dataName}.csv', sep=";")['OutPut_class |T+1|']

    print("******************** Inicio da otimização dos modelos de Classificação ********************")
    print(f'X_train: {X.shape} | Y_train: {Y.shape}')

    # ----------------------------- Otimizando SVM ----------------------------------
    print('------------------------------------ SVM ------------------------------------------')
    SVM_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.arange(0.01, 50, 10),
        'gamma': ['scale', 'auto'],
        'degree': [1, 2, 3, 4],
        'coef0': np.arange(1, 5, 2),
    }
    svm_grid_search = GridSearchCV(estimator = SVC(), param_grid = SVM_grid, cv = 5, n_jobs = -1, verbose = 1, scoring = scoring)
    svm_grid_search.fit(X, Y)
    bestSVM = svm_grid_search.best_estimator_
    print("Melhore Acurácia: ", svm_grid_search.best_score_)
    with open(f'../Results/optimization/classifications/SVM/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestSVM, f)
    
    
    # ----------------------------- Otimizando KNN ----------------------------------
    print('------------------------------------ KNN -----------------------------------------')
    KNN_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 25, 28, 31, 34],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2, 3]
    }
    knn_grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = KNN_grid, cv = 5, n_jobs = -1, verbose = 1, scoring = scoring)
    knn_grid_search.fit(X, Y)
    bestKNN = knn_grid_search.best_estimator_
    print("Melhore Acurácia: ", knn_grid_search.best_score_)
    with open(f'../Results/optimization/classifications/KNN/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestKNN, f)

    # ----------------------------- Otimizando LR ----------------------------------
    print('------------------------------------- LR --------------------------------------------')
    LR_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'None'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'C': [0.1, 1, 10, 100, 1000],
        'max_iter': [50, 100, 500],
        'fit_intercept': [True, False],
        'class_weight': ['balanced', None],
        'warm_start': [True, False]
    }
    lr_grid_search = GridSearchCV(estimator = LogisticRegression(), param_grid = LR_grid, cv = 5, n_jobs = -1, verbose = 1, scoring = scoring)
    lr_grid_search.fit(X, Y)
    bestLR = lr_grid_search.best_estimator_
    print("Melhore Acurácia: ", lr_grid_search.best_score_)
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
        print(type(model).__name__)
        series = pd.Series(name=name, data=model.predict(X_test.values))
        results = pd.concat([results, series], axis=1)


    results.to_csv(f'../Results/test/classification/{dataName}_predictions.csv', sep=';', index=False)
