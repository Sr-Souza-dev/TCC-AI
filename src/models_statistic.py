import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.model_selection as ms
from sklearn.metrics import mean_absolute_error, mean_squared_error
import arch
from arch.__future__ import reindexing

np.seterr(warn='ignore', message='Series.__getitem__ treating keys as positions is deprecated.')


arimaLogs = pd.DataFrame(columns=['autoregressive(p)', 'diferencial(d)', 'media(q)', 'MAE', 'MSE', 'RMSE'])
bestArima = {'autoregressive(p)': 0, 'diferencial(d)': 0, 'media(q)': 0, 'MAE': 100000, 'MSE': 100000, 'RMSE': 100000, 'model': None}

sarimaLogs = pd.DataFrame(columns=['autoregressive(p)', 'diferencial(d)', 'media(q)', 'sazonalidade(P)', 'diferencial(D)', 'media(Q)', 'periodo(S)', 'MAE', 'MSE', 'RMSE'])
bestSarima = {'autoregressive(p)': 0, 'diferencial(d)': 0, 'media(q)': 0, 'sazonalidade(P)': 0, 'diferencial(D)': 0, 'media(Q)': 0, 'periodo(S)': 0, 'MAE': 100000, 'MSE': 100000, 'RMSE': 100000, 'model': None}

garchLogs = pd.DataFrame(columns=['p', 'q', 'MAE', 'MSE', 'RMSE'])
bestGarch = {'p': 0, 'q': 0, 'MAE': 100000, 'MSE': 100000, 'RMSE': 100000, 'model': None}

def saveModelsChanges(model, modelName, values, dataTrain, dataTest):

    Y_pred = []
    if(modelName == 'ARIMA'):
        Y_pred, model = arimaPredict(dataTrain, dataTest, values)
    elif(modelName == 'SARIMA'):
        Y_pred, model = sarimaPredict(dataTrain, dataTest, values)
    elif(modelName == 'GARCH'):
        Y_pred, model = garchPredict(dataTrain, dataTest, values)
    
    mae = mean_absolute_error(dataTest, Y_pred)
    mse = mean_squared_error(dataTest, Y_pred)
    rmse = np.sqrt(mse)

    if(modelName == 'ARIMA'):
        arimaLogs.loc[len(arimaLogs)] = [values[0], values[1], values[2], mae, mse, rmse]
        if(bestArima['MAE'] > mae and bestArima['MSE'] > mse and bestArima['RMSE'] > rmse):
            bestArima['autoregressive(p)'] = values[0]
            bestArima['diferencial(d)'] = values[1]
            bestArima['media(q)'] = values[2]
            bestArima['MAE'] = mae
            bestArima['MSE'] = mse
            bestArima['RMSE'] = rmse
            bestArima['model'] = model
    elif(modelName == 'SARIMA'):
        sarimaLogs.loc[len(sarimaLogs)] = [values[0], values[1], values[2], values[3], values[4], values[5], values[6], mae, mse, rmse]
        if(bestSarima['MAE'] > mae and bestSarima['MSE'] > mse and bestSarima['RMSE'] > rmse):
            bestSarima['autoregressive(p)'] = values[0]
            bestSarima['diferencial(d)'] = values[1]
            bestSarima['media(q)'] = values[2]
            bestSarima['sazonalidade(P)'] = values[3]
            bestSarima['diferencial(D)'] = values[4]
            bestSarima['media(Q)'] = values[5]
            bestSarima['periodo(S)'] = values[6]
            bestSarima['MAE'] = mae
            bestSarima['MSE'] = mse
            bestSarima['RMSE'] = rmse
            bestSarima['model'] = model
    elif(modelName == 'GARCH'):
        garchLogs.loc[len(garchLogs)] = [values[0], values[1], mae, mse, rmse]
        if(bestGarch['MAE'] > mae and bestGarch['MSE'] > mse and bestGarch['RMSE'] > rmse):
            bestGarch['p'] = values[0]
            bestGarch['q'] = values[1]
            bestGarch['MAE'] = mae
            bestGarch['MSE'] = mse
            bestGarch['RMSE'] = rmse
            bestGarch['model'] = model
    

def arimaPredict(dataTrain, dataTest, values):
    Y_pred = []
    model = None
    for i in range(0, len(dataTest)):
        model = sm.tsa.ARIMA(endog=dataTrain, order = (values[0], values[1], values[2])).fit(disp=False)
        prediction = model.predict(start=len(dataTrain), end=len(dataTrain))
        Y_pred.append(prediction[0])  # Adicionar a previsão à lista Y_pred
        dataTrain = np.concatenate((dataTrain, [dataTest[i]]))

    return Y_pred, model

def sarimaPredict(dataTrain, dataTest, values):
    Y_pred = []
    model = None
    for i in range(0, len(dataTest)):
        model = sm.tsa.SARIMAX(endog=dataTrain, order = (values[0], values[1], values[2]), seasonal_order = (values[3], values[4], values[5], values[6])).fit(disp=False)
        prediction = model.predict(start=len(dataTrain), end=len(dataTrain))
        Y_pred.append(prediction[0])  # Adicionar a previsão à lista Y_pred
        dataTrain = np.concatenate((dataTrain, [dataTest[i]]))

    return Y_pred, model

def garchPredict(dataTrain, dataTest, values):
    Y_pred = []
    model = None
    for i in range(0, len(dataTest)):
        model = arch.arch_model(dataTrain, p = int(round(values[0])), q = int(round(values[1])), vol='GARCH', dist='Normal', rescale=True).fit(disp=False)
        prediction = model.forecast(horizon=1)
        Y_pred.append(prediction.mean.values[0][0])  # Adicionar a previsão à lista Y_pred
        dataTrain = np.concatenate((dataTrain, [dataTest[i]]))
    return Y_pred, model


def GetModelsStatisticsOptimized(dataName, size, test_size = 0.4):
    X = pd.read_csv(f'../Data/Cut/statistic/X/Optmz_{size}{dataName}.csv', sep=";")
    Y = pd.read_csv(f'../Data/Cut/statistic/Y/Optmz_{size}{dataName}.csv', sep=";")
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size = test_size, random_state = None, shuffle = False)

    print("******************** Inicio da otimização dos modelos Estatisticos ********************")
    print(f'X_train: {X_train.shape} | X_test: {X_test.shape} | Y_train: {Y_train.shape} | Y_test: {Y_test.shape}')


    # Modelo ARIMA
    p = np.arange(0, 3, 1)
    d = np.arange(0, 2, 1)
    q = np.arange(0, 3, 1)

    print("-------------------- ARIMA --------------------")
    for i in p:
        for j in d:
            for k in q:
                print(f'------ p: {i} | d: {j} | q: {k}')
                saveModelsChanges(None, 'ARIMA', [i, j, k], Y_train['OutPut |T+1|'].ravel(), Y_test['OutPut |T+1|'].ravel())

    print(bestArima)
    arimaLogs.to_csv(f'../Results/optimization/statistics/ARIMA/{dataName}_Logs.csv', sep=';', index=False)


    # Modelo SARIMA
    p = np.arange(0, 2, 1)
    d = np.arange(0, 2, 1)
    q = np.arange(0, 2, 1)

    P = np.arange(1, 2, 1)
    D = np.arange(0, 1, 1)
    Q = np.arange(1, 2, 1)
    S = np.arange(10, 50, 10)

    print("-------------------- SARIMA --------------------")
    for i in p:
        for j in d:
            for k in q:
                for l in P:
                    for m in D:
                        for n in Q:
                            for o in S:
                                print(f'------ p: {i} | d: {j} | q: {k} | P: {l} | D: {m} | Q: {n} | S: {o}')
                                saveModelsChanges(None, 'SARIMA', [i, j, k, l, m, n, o], Y_train['OutPut |T+1|'].ravel(), Y_test['OutPut |T+1|'].ravel())

    print(bestSarima)
    sarimaLogs.to_csv(f'../Results/optimization/statistics/SARIMA/{dataName}_Logs.csv', sep=';', index=False)


    # Modelo GARCH
    p = np.arange(1, 5, 1)
    q = np.arange(1, 5, 1)

    print("-------------------- GARCH --------------------")
    for i in p:
        for j in q:
            print(f'------ p: {i} | q: {j}')
            saveModelsChanges(None, 'GARCH', [i, j], Y_train['OutPut |T+1|'].ravel(), Y_test['OutPut |T+1|'].ravel())


    print(bestGarch)
    garchLogs.to_csv(f'../Results/optimization/statistics/GARCH/{dataName}_Logs.csv', sep=';', index=False)

    return bestArima, bestSarima, bestGarch