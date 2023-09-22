import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def MakeClassificationsLogs(dataName, size):
    regressions     = pd.read_csv(f'../Results/test/regression/{dataName}_predictions_class.csv', sep=';')
    classifications = pd.read_csv(f'../Results/test/classification/{dataName}_predictions.csv', sep=';')
    statistics      = pd.read_csv(f'../Results/test/statistic/{dataName}_predictions_class.csv', sep=';')

    ensambleReg     = pd.read_csv(f'../Results/test/regression/{dataName}_ensamble.csv', sep=';')['class']
    ensambleClass   = pd.read_csv(f'../Results/test/classification/{dataName}_ensamble.csv', sep=';')['class']
    ensambleStat    = pd.read_csv(f'../Results/test/statistic/{dataName}_ensamble.csv', sep=';')['class']

    ensambleReg.name    = 'ensambleReg'
    ensambleClass.name  = 'ensambleClass'
    ensambleStat.name   = 'ensambleStat'

    outputs         = pd.read_csv(f'../Data/Cut/dataset1/Y/Test_{size}{dataName}.csv', sep=";")['OutPut_class |T+1|']

    datas = pd.concat([regressions, classifications, statistics, ensambleReg, ensambleClass, ensambleStat], axis=1)
    #datas = pd.concat([regressions, classifications, statistics], axis=1)


    logs = pd.DataFrame(columns=['model', 'accuracy', 'f1', 'truePositives', 'trueNegatives', 'falsePositives', 'falseNegatives'])
    for column in datas.columns:
        accuracy = accuracy_score(outputs.ravel(), datas[column].ravel())
        f1 = f1_score(outputs.ravel(), datas[column].ravel())
        matrix = confusion_matrix(outputs.ravel(), datas[column].ravel())
        logs = pd.concat([logs, pd.DataFrame({'model': [column], 'accuracy': [accuracy], 'f1': [f1], 'truePositives': [matrix[0][0]], 'trueNegatives': [matrix[1][1]], 'falsePositives': [matrix[0][1]], 'falseNegatives': [matrix[1][0]]})], axis=0)
    
    logs.to_csv(f'../Results/test/logs/{dataName}_logs.csv', sep=';', index=False)