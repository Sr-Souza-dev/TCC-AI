from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sklearn.model_selection as ms
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

import warnings
warnings.filterwarnings("ignore")

namesM = ['SVM', 'KNN', 'LR']

svmLogs = pd.DataFrame(columns=['kernel', 'regularization', 'gamma', 'degree', 'coef0', 'accuracy', 'f1_score', 'True Positive', 'False Positive', 'False Negative', 'True Negative'])
bestSVM = {'kernel': 'linear', 'regularization': 0.01, 'gamma': 'default', 'degree': 'default', 'coef0': 'default', 'accuracy': 0, 'f1_score': 0, 'True Positive': 0, 'False Positive': 0, 'False Negative': 0, 'True Negative': 0}

knnLogs = pd.DataFrame(columns=['neighbors', 'weights', 'algorithm', 'distance', 'accuracy', 'f1_score', 'True Positive', 'False Positive', 'False Negative', 'True Negative'])
bestKNN = {'neighbors': 0, 'weights': 'uniform', 'algorithm': 'auto', 'distance': 'euclidean', 'accuracy': 0, 'f1_score': 0, 'True Positive': 0, 'False Positive': 0, 'False Negative': 0, 'True Negative': 0}

lrLogs = pd.DataFrame(columns=['penalty', 'solver', 'regularization', 'max_iter', 'fit_intercept', 'class_weight', 'warm_start', 'accuracy', 'f1_score', 'True Positive', 'False Positive', 'False Negative', 'True Negative'])
bestLR = {'penalty': 'l1', 'solver': 'newton-cg', 'regularization': 0.01, 'max_iter': 100, 'fit_intercept': True, 'class_weight': 'balanced', 'warm_start': True, 'accuracy': 0, 'f1_score': 0, 'True Positive': 0, 'False Positive': 0, 'False Negative': 0, 'True Negative': 0}

def saveModelChanges(model, modelName, values, X_train, X_test, Y_train, Y_test):
    model.fit(X_train.values, Y_train['OutPut_class |T+1|'].ravel())
    Y_pred = model.predict(X_test.values)

    accuracy    = accuracy_score(Y_test['OutPut_class |T+1|'].ravel(), Y_pred)
    f1          = f1_score(Y_test['OutPut_class |T+1|'].ravel(), Y_pred)
    confMatrix  = confusion_matrix(Y_test['OutPut_class |T+1|'].ravel(), Y_pred)

    if(modelName == namesM[0]):
        saveBestModelSVM(accuracy, f1, model, values, confMatrix)
        svmLogs.loc[len(svmLogs)] = values + [accuracy, f1, confMatrix[0][0], confMatrix[1][0], confMatrix[0][1], confMatrix[1][1]]
    elif(modelName == namesM[1]):
        saveBestModelKNN(accuracy, f1, model, values, confMatrix)
        knnLogs.loc[len(knnLogs)] = values + [accuracy, f1, confMatrix[0][0], confMatrix[1][0], confMatrix[0][1], confMatrix[1][1]]
    elif(modelName == namesM[2]):
        saveBestModelLR(accuracy, f1, model, values, confMatrix)
        lrLogs.loc[len(lrLogs)] = values + [accuracy, f1, confMatrix[0][0], confMatrix[1][0], confMatrix[0][1], confMatrix[1][1]]


    
def saveBestModelSVM(accuracy, f1, model, values, confMatrix):
    if(accuracy > bestSVM['accuracy'] and f1 > bestSVM['f1_score']):
        bestSVM['kernel'] = values[0]
        bestSVM['regularization'] = values[1]
        bestSVM['gamma'] = values[2]
        bestSVM['degree'] = values[3]
        bestSVM['coef0'] = values[4]
        bestSVM['accuracy'] = accuracy
        bestSVM['f1_score'] = f1
        bestSVM['True Positive'] = confMatrix[0][0]
        bestSVM['False Positive'] = confMatrix[1][0]
        bestSVM['False Negative'] = confMatrix[0][1]
        bestSVM['True Negative'] = confMatrix[1][1]
        bestSVM['model'] = model

def saveBestModelKNN(accuracy, f1, model, values, confMatrix):
    if(accuracy > bestKNN['accuracy'] and f1 > bestKNN['f1_score']):
        bestKNN['neighbors'] = values[0]
        bestKNN['weights'] = values[1]
        bestKNN['algorithm'] = values[2]
        bestKNN['distance'] = values[3]
        bestKNN['accuracy'] = accuracy
        bestKNN['f1_score'] = f1
        bestKNN['True Positive'] = confMatrix[0][0]
        bestKNN['False Positive'] = confMatrix[1][0]
        bestKNN['False Negative'] = confMatrix[0][1]
        bestKNN['True Negative'] = confMatrix[1][1]
        bestKNN['model'] = model

def saveBestModelLR(accuracy, f1, model, values, confMatrix):
    if(accuracy > bestLR['accuracy'] and f1 > bestLR['f1_score']):
        bestLR['penalty'] = values[0]
        bestLR['solver'] = values[1]
        bestLR['regularization'] = values[2]
        bestLR['max_iter'] = values[3]
        bestLR['fit_intercept'] = values[4]
        bestLR['class_weight'] = values[5]
        bestLR['warm_start'] = values[6]
        bestLR['accuracy'] = accuracy
        bestLR['f1_score'] = f1
        bestLR['True Positive'] = confMatrix[0][0]
        bestLR['False Positive'] = confMatrix[1][0]
        bestLR['False Negative'] = confMatrix[0][1]
        bestLR['True Negative'] = confMatrix[1][1]
        bestLR['model'] = model

def GetModelsClassificationOptimized(dataName, size,  test_size = 0.4):
    X = pd.read_csv(f'../Data/Cut/dataset1/X/Optmz_{size}{dataName}.csv', sep=";")
    Y = pd.read_csv(f'../Data/Cut/dataset1/Y/Optmz_{size}{dataName}.csv', sep=";")
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size = test_size, random_state = None, shuffle = False)

    print("******************** Inicio da otimização dos modelos de Classificação ********************")
    print(f'X_train: {X_train.shape} | X_test: {X_test.shape} | Y_train: {Y_train.shape} | Y_test: {Y_test.shape}')

    # ----------------------------- Otimizando SVM ----------------------------------
    print('------------------------------------ SVM ------------------------------------------')
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    regularizations = np.arange(0.01, 50, 5)
    gammas = np.arange(0.01, 5, 1)                 # para kernels rbf, poly e sigmoid
    degrees = np.arange(1, 4, 1)                   # para kernel poly
    coef0s = np.arange(1, 5, 1)                    # para kernel poly e sigmoid

    for kernel in kernels:
        for regularization in regularizations:
            try:
                if kernel == 'linear':
                    print(f'------- kernel: {kernel} | Regularization: {regularization}')
                    model = SVC(kernel=kernel, C=regularization)
                    values = [kernel, regularization, 'default', 'default', 'default']
                    saveModelChanges(model, namesM[0], values, X_train, X_test, Y_train, Y_test)
                elif kernel == 'poly':
                    for gamma in gammas:
                        for degree in degrees:
                            for coef0 in coef0s:
                                print(f'------- kernel: {kernel} | regularization: {regularization} | gamma: {gamma} | degree: {degree} | coef0: {coef0}')
                                model = SVC(kernel=kernel, C=regularization, gamma=gamma, degree=degree, coef0=coef0)
                                values = [kernel, regularization, gamma, degree, coef0]
                                saveModelChanges(model, namesM[0], values, X_train, X_test, Y_train, Y_test)
                elif kernel == 'rbf':
                    for gamma in gammas:
                        print(f'------- kernel: {kernel} | regularization: {regularization} | gamma: {gamma}')
                        model = SVC(kernel=kernel, C=regularization, gamma=gamma)
                        values = [kernel, regularization, gamma, 'default', 'default']
                        saveModelChanges(model, namesM[0], values, X_train, X_test, Y_train, Y_test)
                elif kernel == 'sigmoid':
                    for gamma in gammas:
                        for coef0 in coef0s:
                            print(f'------- kernel: {kernel} | regularization: {regularization} | gamma: {gamma} | coef0: {coef0}')
                            model = SVC(kernel=kernel, C=regularization, gamma=gamma, coef0=coef0)
                            values = [kernel, regularization, gamma, 'default', coef0]
                            saveModelChanges(model, namesM[0], values, X_train, X_test, Y_train, Y_test)
            except:
                print("Erro ao Otimizar o Modelo SVM com os parametros informados")
                continue

    print(bestSVM)
    svmLogs.loc[0] = [bestSVM['kernel'], bestSVM['regularization'], bestSVM['gamma'], bestSVM['degree'], bestSVM['coef0'], bestSVM['accuracy'], bestSVM['f1_score'], bestSVM['True Positive'], bestSVM['False Positive'], bestSVM['False Negative'], bestSVM['True Negative']]
    svmLogs.to_csv(f'../Results/optimization/classifications/SVM/{dataName}_Logs.csv', sep=';', index=False)
    with open(f'../Results/optimization/classifications/SVM/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestSVM['model'], f)
    
    
    # ----------------------------- Otimizando KNN ----------------------------------
    print('------------------------------------ KNN -----------------------------------------')
    neighbors = np.arange(1, 20, 1)
    weights = ['uniform', 'distance']
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    distances = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

    for neighbor in neighbors:
        for weight in weights:
            for algorithm in algorithms:
                for distance in distances:
                    try:
                        print(f'------- neighbors: {neighbor} | weights: {weight} | algorithm: {algorithm} | distance: {distance}')
                        model = KNeighborsClassifier(n_neighbors=neighbor, weights=weight, algorithm=algorithm, metric=distance)
                        values = [neighbor, weight, algorithm, distance]
                        saveModelChanges(model, namesM[1], values, X_train, X_test, Y_train, Y_test)
                    except:
                        print("Erro ao Otimizar o Modelo KNN com os parametros informados")
                        continue
    print(bestKNN)
    knnLogs.loc[0] = [bestKNN['neighbors'], bestKNN['weights'], bestKNN['algorithm'], bestKNN['distance'], bestKNN['accuracy'], bestKNN['f1_score'], bestKNN['True Positive'], bestKNN['False Positive'], bestKNN['False Negative'], bestKNN['True Negative']]
    knnLogs.to_csv(f'../Results/optimization/classifications/KNN/{dataName}_Logs.csv', sep=';', index=False)
    with open(f'../Results/optimization/classifications/KNN/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestKNN['model'], f)

    # ----------------------------- Otimizando LR ----------------------------------
    print('------------------------------------- LR --------------------------------------------')
    penalty = ['l2']
    solvers = ['lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg']
    regularizations = np.arange(0.01, 50, 0.5)
    max_iterations = np.arange(100, 1500, 100)
    fit_intercepts = [True, False]
    class_weights = ['balanced', None]
    warm_starts = [True, False]


    for pen in penalty:
        for solver in solvers:
            for regularization in regularizations:
                for max_iter in max_iterations:
                    for fit_intercept in fit_intercepts:
                        for class_weight in class_weights:
                            for warm_start in warm_starts:
                                try:
                                    print(f'------- penalty: {pen} | solver: {solver} | regularization: {regularization} | max_iter: {max_iter} | fit_intercept: {fit_intercept} | class_weight: {class_weight} | warm_start: {warm_start}')
                                    model = LogisticRegression(penalty=pen, solver=solver, C=regularization, max_iter=max_iter, fit_intercept=fit_intercept, class_weight=class_weight, warm_start=warm_start)
                                    values = [pen, solver, regularization, max_iter, fit_intercept, class_weight, warm_start]
                                    saveModelChanges(model, namesM[2], values, X_train, X_test, Y_train, Y_test)
                                except:
                                    print("Erro ao Otimizar o Modelo LR com os parametros informados")
                                    continue
    
    print(bestLR)
    lrLogs.loc[0] = [bestLR['penalty'], bestLR['solver'], bestLR['regularization'], bestLR['max_iter'], bestLR['fit_intercept'], bestLR['class_weight'], bestLR['warm_start'], bestLR['accuracy'], bestLR['f1_score'], bestLR['True Positive'], bestLR['False Positive'], bestLR['False Negative'], bestLR['True Negative']]
    lrLogs.to_csv(f'../Results/optimization/classifications/LR/{dataName}_Logs.csv', sep=';', index=False)
    with open(f'../Results/optimization/classifications/LR/{dataName}_model.pkl', 'wb') as f:
        pickle.dump(bestLR['model'], f)

    return bestSVM, bestKNN, bestLR


def GetModelsClassification(dataName):
    with open(f'../Results/optimization/classifications/SVM/{dataName}_model.pkl', 'rb') as f:
        SVM = pickle.load(f)
    with open(f'../Results/optimization/classifications/KNN/{dataName}_model.pkl', 'rb') as f:
        KNN = pickle.load(f)
    with open(f'../Results/optimization/classifications/LR/{dataName}_model.pkl', 'rb') as f:
        LR = pickle.load(f)

    return SVM, KNN, LR
