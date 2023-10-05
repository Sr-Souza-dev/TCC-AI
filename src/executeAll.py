import time
from data_generate import Generate
from data_selection import Selection
from data_cut import Cut
from models_classification import GetModelsClassificationOptimized
from models_statistic import GetModelsStatisticsOptimized
from models_regression import GetModelsRegressionOptimized
from models_classification import GetModelsClassification
from models_statistic import GetModelsStatistics
from models_regression import GetModelsRegression
from models_statistic import GetStatisticPredictions
from models_classification import GetClassificationPredictions
from models_regression import GetRegressionPredictions
import pandas as pd
from strategy import GetEnsambles
from models_buying import GetModelPrediction
from analyze import MakeClassificationsLogs
from analyze_economy import GetEconomyAnalyze
import warnings

def RunSolution(dataName, outputName, setDivision):
    warnings.filterwarnings("ignore")
    try:
        total_time = time.time()

        # --------------------------- Gera a base de Dados ---------------------------
        generate_Time = time.time()
        dataShape, outShape = Generate(dataName, outputName)    # Gera a base de dados
        d1Shape, d2Shape, outShape = Selection(dataName)        # Seleciona os dados gerando o dataset1 e dataset2
        Cut(dataName, setDivision)                              # divide a base de dados em otimização, treino e teste
        generate_Time = time.time() - generate_Time

        # --------------------------- Obtém os modelos Otimizados ---------------------------
        optmized_time = time.time()
        SVM, KNN, LR = GetModelsClassificationOptimized(dataName, setDivision[1])           # Obtém os modelos de classificação otimizados
        ARIMA, SARIMA, GARCH = GetModelsStatisticsOptimized(dataName, setDivision[0])       # Obtém os modelos de estatística otimizados
        LSTM, MLP, RNN = GetModelsRegressionOptimized(dataName, setDivision[1])             # Obtém os modelos de regressão otimizados

        # ------------------------ Recupera os modelos já otimizados --------------------------
        SVM, KNN, LR = GetModelsClassification(dataName)        # Obtém os modelos de classificação
        ARIMA, SARIMA, GARCH = GetModelsStatistics(dataName)    # Obtém os modelos de estatística
        LSTM, MLP, RNN = GetModelsRegression(dataName)          # Obtém os modelos de regressão
        optmized_time = time.time() - optmized_time

        # --------------------------- Treina os modelos ---------------------------
        train_time = time.time()
        Y_Train_statistic = pd.read_csv(f'../Data/Cut/statistic/Y/Train_{setDivision[1]}{dataName}.csv', sep=";")['OutPut |T+1|']   # Obtém os dados de treino para os modelos de estatística
        Y_Test_statistic  = pd.read_csv(f'../Data/Cut/statistic/Y/Test_{setDivision[2]}{dataName}.csv', sep=";")['OutPut |T+1|']    # Obtém os dados de teste para os modelos de estatística
        X_Test_dataset1  = pd.read_csv(f'../Data/Cut/dataset1/X/Test_{setDivision[2]}{dataName}.csv', sep=";")                      # Obtém os dados de teste para os modelos de classificação
        X_Train_dataset2 = pd.read_csv(f'../Data/Cut/dataset2/X/Train_{setDivision[1]}{dataName}.csv', sep=";")                     # Obtém os dados de treino para os modelos de regressão
        Y_Train_dataset2 = pd.read_csv(f'../Data/Cut/dataset2/Y/Train_{setDivision[1]}{dataName}.csv', sep=";")['OutPut |T+1|']     # Obtém os dados de treino para os modelos de regressão
        X_Test_dataset2  = pd.read_csv(f'../Data/Cut/dataset2/X/Test_{setDivision[2]}{dataName}.csv', sep=";")                      # Obtém os dados de teste para os modelos de regressão
        Y_Test_dataset2  = pd.read_csv(f'../Data/Cut/dataset2/Y/Test_{setDivision[2]}{dataName}.csv', sep=";")['OutPut |T+1|']      # Obtém os dados de teste para os modelos de regressão

        ClassificationModels = [SVM, KNN, LR]           # Lista de modelos de classificação
        ClassificationNames  = ['SVM', 'KNN', 'LR']     # Lista de nomes dos modelos de classificação
        RegressionModels = [LSTM, MLP, RNN]             # Lista de modelos de regressão
        RegressionNames  = ['LSTM', 'MLP', 'RNN']       # Lista de nomes dos modelos de regressão

        GetClassificationPredictions(dataName, ClassificationModels, ClassificationNames, X_Test_dataset1)                                              # Obtém as predições dos modelos de classificação
        GetRegressionPredictions(dataName, RegressionNames, RegressionModels, X_Test_dataset2, Y_Test_dataset2, X_Train_dataset2, Y_Train_dataset2)     # Obtém as predições dos modelos de regressão
        GetStatisticPredictions(dataName, Y_Train_statistic.ravel(), Y_Test_statistic.ravel(), window=100)                                              # Obtém as predições dos modelos de estatística
        train_time = time.time() - train_time

        # --------------------------- Obtendo ensambles ---------------------------
        ensamble_time = time.time()
        GetEnsambles(dataName, setDivision[2], setDivision[1])      # Obtém os ensambles
        GetModelPrediction(dataName, setDivision[1])                # Obtém as predições do modelo de compra
        ensamble_time = time.time() - ensamble_time

        # --------------------------- Obtendo resultados ---------------------------
        MakeClassificationsLogs(dataName, setDivision[2])           # Obtém os logs de classificação
        GetEconomyAnalyze(dataName, setDivision[2])                 # Obtém os logs de economia

        # Gera uma string com todas informações de tempo e tamanho do dataset utilizado para a entrada dataName
        total_time = time.time() - total_time
        info = f""" -------------------- {dataName} -------------------- 
                | Data Shape: {dataShape} 
                | Output Shape: {outShape} 
                | Dataset1 Shape: {d1Shape} 
                | Dataset2 Shape: {d2Shape} 
                | Generate Time: {generate_Time} 
                | Optmized Time: {optmized_time} 
                | Train Time: {train_time} 
                | Ensambles Time: {ensamble_time} 
                | Total Time: {total_time} 
                ------------------------------------------------------- \n\n"""
        return info
    except Exception as e:
        return f""" -------------------- {dataName} --------------------
                | Erro: {e}
                ------------------------------------------------------- \n\n"""