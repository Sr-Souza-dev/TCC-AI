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
from ensambles import GetEnsambles
from models_buying import GetModelPrediction
from analyze import MakeClassificationsLogs
from analyze_economy import GetEconomyAnalyze
import warnings
import git

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals.")
warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated.")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge.")

def commit_and_push(commit_message="Automated commit", branch="main"):
    try:
        repo = git.Repo("../")
        repo.git.add("--all")
        repo.index.commit(commit_message)
        origin = repo.remote(name="origin")
        origin.push(refspec=f"{branch}:{branch}")
        print("Commit e push bem-sucedidos.")
    except Exception as e:
        print(f"Erro ao realizar commit e push: {e}")



def getDatabase(dataName, outputName, setDivision):
    try:
        # --------------------------- Gera a base de Dados ---------------------------
        print(f"    | Etapa 1 {dataName} - Gerando base de dados")
        generate_Time = time.time()
        dataShape, outShape = Generate(dataName, outputName)    # Gera a base de dados
        d1Shape, d2Shape, outShape = Selection(dataName)        # Seleciona os dados gerando o dataset1 e dataset2
        Cut(dataName, setDivision)                              # divide a base de dados em otimização, treino e teste
        generate_Time = time.time() - generate_Time
        print(f"    | Etapa 1 {dataName} - Time: {generate_Time}")
    except Exception as e:
        print(f"    | ERRO - Etapa 1 {dataName} (gerando base de dados)")
        print(f"        - {e}")

def getOptmizedModels(dataName, setDivision):
    try:
        # --------------------------- Obtém os modelos Otimizados ---------------------------
        print(f"    | Etapa 2 {dataName} - Obtendo modelos otimizados!")
        optmized_time = time.time()
        print(f"        - Etapa 2.1 {dataName} - Obtendo modelos de classificação otimizados")
        SVM, KNN, LR = GetModelsClassificationOptimized(dataName, setDivision[1])           # Obtém os modelos de classificação otimizados
        print(f"        - Etapa 2.2 {dataName} - Obtendo modelos estatisticos otimizados")
        ARIMA, SARIMA, GARCH = GetModelsStatisticsOptimized(dataName, setDivision[0])       # Obtém os modelos de estatística otimizados
        print(f"        - Etapa 2.3 {dataName} - Obtendo modelos de regressão otimizados")
        LSTM, MLP, RNN = GetModelsRegressionOptimized(dataName, setDivision[1])             # Obtém os modelos de regressão otimizados
        optmized_time = time.time() - optmized_time
        print(f"    | Etapa 2 {dataName} - Time: {optmized_time}")
    except Exception as e:
        print(f"    | ERRO - Etapa 2 {dataName} (Obtendo modelos otimizados)")
        print(f"        - {e}")

def trainModels(dataName, setDivision):
    try:
        # ------------------------ Recupera os modelos já otimizados --------------------------
        print(f"    | Etapa 3 {dataName} - Obtendo modelos já otimizados!")
        SVM, KNN, LR = GetModelsClassification(dataName)        # Obtém os modelos de classificação
        LSTM, MLP, RNN = GetModelsRegression(dataName)          # Obtém os modelos de regressão

        # --------------------------- Treina os modelos ---------------------------
        print(f"    | Etapa 4 {dataName} - Treinando modelos!")
        train_time = time.time()
        Y_Train_statistic = pd.read_csv(f'../Data/Cut/statistic/Y/Train_{setDivision[1]}{dataName}.csv', sep=";")['OutPut |T+1|']   # Obtém os dados de treino para os modelos de estatística
        Y_Test_statistic  = pd.read_csv(f'../Data/Cut/statistic/Y/Test_{dataName}.csv', sep=";")['OutPut |T+1|']    # Obtém os dados de teste para os modelos de estatística
        X_Test_dataset1  = pd.read_csv(f'../Data/Cut/dataset1/X/Test_{dataName}.csv', sep=";")                      # Obtém os dados de teste para os modelos de classificação
        X_Train_dataset2 = pd.read_csv(f'../Data/Cut/dataset2/X/Train_{setDivision[1]}{dataName}.csv', sep=";")                     # Obtém os dados de treino para os modelos de regressão
        Y_Train_dataset2 = pd.read_csv(f'../Data/Cut/dataset2/Y/Train_{setDivision[1]}{dataName}.csv', sep=";")['OutPut |T+1|']     # Obtém os dados de treino para os modelos de regressão
        X_Test_dataset2  = pd.read_csv(f'../Data/Cut/dataset2/X/Test_{dataName}.csv', sep=";")                      # Obtém os dados de teste para os modelos de regressão
        Y_Test_dataset2  = pd.read_csv(f'../Data/Cut/dataset2/Y/Test_{dataName}.csv', sep=";")['OutPut |T+1|']      # Obtém os dados de teste para os modelos de regressão

        ClassificationModels = [SVM, KNN, LR]           # Lista de modelos de classificação
        ClassificationNames  = ['SVM', 'KNN', 'LR']     # Lista de nomes dos modelos de classificação
        RegressionModels = [LSTM, MLP, RNN]             # Lista de modelos de regressão
        RegressionNames  = ['LSTM', 'MLP', 'RNN']       # Lista de nomes dos modelos de regressão

        print(f"        - Etapa 4.1 {dataName} - Treinando modelos de classificação")
        GetClassificationPredictions(dataName, ClassificationModels, ClassificationNames, X_Test_dataset1)                                              # Obtém as predições dos modelos de classificação
        print(f"        - Etapa 4.2 {dataName} - Treinando modelos de regressão")
        GetRegressionPredictions(dataName, RegressionNames, RegressionModels, X_Test_dataset2, Y_Test_dataset2, X_Train_dataset2, Y_Train_dataset2)     # Obtém as predições dos modelos de regressão
        print(f"        - Etapa 4.3 {dataName} - Treinando modelos de estatística")
        GetStatisticPredictions(dataName, Y_Train_statistic.ravel(), Y_Test_statistic.ravel(), window=100)                                              # Obtém as predições dos modelos de estatística
        train_time = time.time() - train_time
        print(f"    | Etapa 4 {dataName} - Time: {train_time}")
    except Exception as e:
        print(f"    | ERRO - Etapa 4 {dataName} (Treinando modelos)")
        print(f"        - {e}")

    
def getEnsambles(dataName, setDivision):
    try:
        # --------------------------- Obtendo ensambles ---------------------------
        print(f"    | Etapa 5 {dataName} - Obtendo ensambles!")
        ensamble_time = time.time()
        GetEnsambles(dataName, setDivision[2], setDivision[1])      # Obtém os ensambles
        GetModelPrediction(dataName, setDivision[1])                # Obtém as predições do modelo de compra
        ensamble_time = time.time() - ensamble_time
        print(f"    | Etapa 5 {dataName} - Time: {ensamble_time}")
    except Exception as e:
        print(f"    | ERRO - Etapa 5 {dataName} (Obtendo ensambles)")
        print(f"        - {e}")

def getResults(dataName, setDivision):
    try:
        # --------------------------- Obtendo resultados ---------------------------
        print(f"    | Etapa 6 {dataName} - Obtendo resultados!")
        MakeClassificationsLogs(dataName, setDivision[2])           # Obtém os logs de classificação
        GetEconomyAnalyze(dataName, setDivision[2])                 # Obtém os logs de economia
    except Exception as e:
        print(f"    | ERRO - Etapa 6 {dataName} (Obtendo resultados)")
        print(f"        - {e}")



def RunSolution(dataName, outputName, setDivision):
    warnings.filterwarnings("ignore")
    try:
        total_time = time.time()

        # --------------------------- Gera a base de Dados ---------------------------
        generate_Time = time.time()
        getDatabase(dataName, outputName, setDivision)                        # divide a base de dados em otimização, treino e teste
        generate_Time = time.time() - generate_Time

        # --------------------------- Obtém os modelos Otimizados ---------------------------
        optmized_time = time.time()
        getOptmizedModels(dataName, setDivision)
        optmized_time = time.time() - optmized_time

        # --------------------------- Treina os modelos ---------------------------
        train_time = time.time()
        trainModels(dataName, setDivision)                                             # Obtém as predições dos modelos de estatística
        train_time = time.time() - train_time

        # --------------------------- Obtendo ensambles ---------------------------
        ensamble_time = time.time()
        getEnsambles(dataName, setDivision)
        ensamble_time = time.time() - ensamble_time

        # --------------------------- Obtendo resultados ---------------------------
        getResults(dataName, setDivision)

        # Gera uma string com todas informações de tempo e tamanho do dataset utilizado para a entrada dataName
        total_time = time.time() - total_time
        info = f""" -------------------- {dataName} -------------------- 
                | Generate Time: {generate_Time} 
                | Optmized Time: {optmized_time} 
                | Train Time: {train_time} 
                | Ensambles Time: {ensamble_time} 
                | Total Time: {total_time} 
                ------------------------------------------------------- \n\n"""
        return info
    except Exception as e:
        print(f"    | Erro na execução do {dataName}")
        print(e)
        return f""" -------------------- {dataName} --------------------
                | Erro: {e}
                ------------------------------------------------------- \n\n"""