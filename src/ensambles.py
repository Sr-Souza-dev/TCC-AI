from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

names = ['regressions', 'classifications', 'statistics']

bestRegressionsWeights = []
bestRegressionMetrics = [0,0]
bestEnsambleRegression = pd.DataFrame({})

bestClassificationsWeights = []
bestClassificationMetrics = [0,0]
bestEnsambleClassification = pd.DataFrame({})

bestStatisticsWeights = []
bestStatisticMetrics = [0,0]
bestEnsambleStatistic = pd.DataFrame({})

def clearDefault():
    global bestRegressionMetrics
    global bestEnsambleRegression
    global bestClassificationMetrics
    global bestEnsambleClassification
    global bestStatisticMetrics
    global bestEnsambleStatistic

    bestRegressionMetrics = [0,0]
    bestEnsambleRegression = pd.DataFrame({})

    bestClassificationMetrics = [0,0]
    bestEnsambleClassification = pd.DataFrame({})

    bestStatisticMetrics = [0,0]
    bestEnsambleStatistic = pd.DataFrame({})


def calculateHits(values, weights):
    denominador = sum(weights)
    return sum(values * weights)/denominador

def getBestWeights(data, weights, name, output):
    global bestRegressionsWeights
    global bestStatisticsWeights
    global bestClassificationsWeights
    global bestEnsambleClassification
    global bestEnsambleRegression
    global bestEnsambleStatistic

    ensamble = pd.DataFrame(columns=['up', 'down', 'class'])
    previsions = []

    for line in range(data.shape[0]):
        line = data.loc[line].values
        hits = calculateHits(line, weights)
        ensamble = ensamble.dropna(axis=1, how='all')
        ensamble = pd.concat([ensamble, pd.DataFrame({'up': [hits], 'down': [1-hits], 'class': [1 if hits>0.6 else 0]})], axis=0)
        if hits > 0.6:
            previsions.append(1)
        else:
            previsions.append(0)

    accuracy = accuracy_score(output, previsions)
    f1 = f1_score(output, previsions)

    if name == names[0] and accuracy > bestRegressionMetrics[0]:
        bestRegressionMetrics[0] = accuracy
        bestRegressionMetrics[1] = f1
        bestRegressionsWeights = weights
        bestEnsambleRegression = ensamble

    elif name == names[1] and accuracy > bestClassificationMetrics[0]:
        bestClassificationMetrics[0] = accuracy
        bestClassificationMetrics[1] = f1
        bestClassificationsWeights = weights
        bestEnsambleClassification = ensamble
        
    elif name == names[2] and accuracy > bestStatisticMetrics[0]:
        bestStatisticMetrics[0] = accuracy
        bestStatisticMetrics[1] = f1
        bestStatisticsWeights = weights
        bestEnsambleStatistic = ensamble
       
    

def GetEnsambles(dataName, testSize = 0.2, trainSize = 0.7):
    regressions     = pd.read_csv(f'../Results/train/regression/{dataName}_predictions_class.csv', sep=';')
    classifications = pd.read_csv(f'../Results/train/classification/{dataName}_predictions.csv', sep=';')
    statistics      = pd.read_csv(f'../Results/train/statistic/{dataName}_predictions_class.csv', sep=';')
    outputs         = pd.read_csv(f'../Data/Cut/dataset1/Y/Train_{trainSize}{dataName}.csv', sep=";")['OutPut_class |T+1|']

    # print("****************************** Obtendo os melhores Ensambles ******************************")
    # print("Regressions Shape:     ", regressions.shape)
    # print("Classifications Shape: ", classifications.shape)
    # print("Statistics Shape:      ", statistics.shape)
    # print("Outputs Shape:         ", outputs.shape)

    datas = [regressions, classifications, statistics]
    

    weights = np.arange(0.0, 1.01, 0.25)
    for idx in range(len(datas)):
        data = datas[idx]
        name = names[idx]
        for m1 in weights:
            for m2 in weights:
                for m3 in weights:
                    getBestWeights(data, [m1, m2, m3], name, outputs.ravel())

    bestEnsambleRegression.to_csv(f'../Results/train/regression/{dataName}_ensamble.csv', sep=';', index=False)
    bestEnsambleClassification.to_csv(f'../Results/train/classification/{dataName}_ensamble.csv', sep=';', index=False)
    bestEnsambleStatistic.to_csv(f'../Results/train/statistic/{dataName}_ensamble.csv', sep=';', index=False)

    # print("------------------------- Esamble de Regrssão -----------------------")
    # print(f'Melhores Pesos de Regressão: {bestRegressionsWeights}')
    # print(f'Acurácia: {bestRegressionMetrics[0]}, F1: {bestRegressionMetrics[1]}')

    # print("------------------------- Esamble de Classificação -----------------------")
    # print(f'Melhores Pesos de Classificação: {bestClassificationsWeights}')
    # print(f'Acurácia: {bestClassificationMetrics[0]}, F1: {bestClassificationMetrics[1]}')

    # print("------------------------- Esamble de Estatística -----------------------")
    # print(f'Melhores Pesos de Estatística: {bestStatisticsWeights}')
    # print(f'Acurácia: {bestStatisticMetrics[0]}, F1: {bestStatisticMetrics[1]}')

    regressions     = pd.read_csv(f'../Results/test/regression/{dataName}_predictions_class.csv', sep=';')
    classifications = pd.read_csv(f'../Results/test/classification/{dataName}_predictions.csv', sep=';')
    statistics      = pd.read_csv(f'../Results/test/statistic/{dataName}_predictions_class.csv', sep=';')
    outputs         = pd.read_csv(f'../Data/Cut/dataset1/Y/Test_{dataName}.csv', sep=";")['OutPut_class |T+1|']

    # print("****************************** Testando os Ensambles ******************************")
    # print("Regressions Shape:     ", regressions.shape)
    # print("Classifications Shape: ", classifications.shape)
    # print("Statistics Shape:      ", statistics.shape)
    # print("Outputs Shape:         ", outputs.shape)

    clearDefault()

    getBestWeights(regressions, bestRegressionsWeights, names[0], outputs.ravel())
    getBestWeights(classifications, bestClassificationsWeights, names[1], outputs.ravel())
    getBestWeights(statistics, bestStatisticsWeights, names[2], outputs.ravel())

    bestEnsambleRegression.to_csv(f'../Results/test/regression/{dataName}_ensamble.csv', sep=';', index=False)
    bestEnsambleClassification.to_csv(f'../Results/test/classification/{dataName}_ensamble.csv', sep=';', index=False)
    bestEnsambleStatistic.to_csv(f'../Results/test/statistic/{dataName}_ensamble.csv', sep=';', index=False)


    # print("------------------------- Esamble de Regrssão -----------------------")
    # print(f'Melhores Pesos de Regressão: {bestRegressionsWeights}')
    # print(f'Acurácia: {bestRegressionMetrics[0]}, F1: {bestRegressionMetrics[1]}')

    # print("------------------------- Esamble de Classificação -----------------------")
    # print(f'Melhores Pesos de Classificação: {bestClassificationsWeights}')
    # print(f'Acurácia: {bestClassificationMetrics[0]}, F1: {bestClassificationMetrics[1]}')

    # print("------------------------- Esamble de Estatística -----------------------")
    # print(f'Melhores Pesos de Estatística: {bestStatisticsWeights}')
    # print(f'Acurácia: {bestStatisticMetrics[0]}, F1: {bestStatisticMetrics[1]}')

