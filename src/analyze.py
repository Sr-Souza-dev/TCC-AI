import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def MakeClassificationsLogs(dataName, size):
    regressions     = pd.read_csv(f'../Results/test/regression/{dataName}_predictions_class.csv', sep=';')
    classifications = pd.read_csv(f'../Results/test/classification/{dataName}_predictions.csv', sep=';')
    statistics      = pd.read_csv(f'../Results/test/statistic/{dataName}_predictions_class.csv', sep=';')

    ensambleReg     = pd.read_csv(f'../Results/test/regression/{dataName}_ensamble.csv', sep=';')['class']
    ensambleClass   = pd.read_csv(f'../Results/test/classification/{dataName}_ensamble.csv', sep=';')['class']
    ensambleStat    = pd.read_csv(f'../Results/test/statistic/{dataName}_ensamble.csv', sep=';')['class']

    buying          = pd.read_csv(f'../Results/test/ensamble1/{dataName}.csv', sep=';')['class']
    buying.name     = 'buying'
    buying2         = pd.read_csv(f'../Results/test/ensamble2/{dataName}.csv', sep=';')['class']
    buying2.name    = 'buying2'

    buyingSVR       = pd.read_csv(f'../Results/test/ensamble1/{dataName}_SVR.csv', sep=';')['class']
    buyingSVR.name  = 'buying_SVR'
    buying2SVR      = pd.read_csv(f'../Results/test/ensamble2/{dataName}_SVR.csv', sep=';')['class']
    buying2SVR.name = 'buying2SVR'

    ensambleReg.name    = 'Ensamble Regression'
    ensambleClass.name  = 'Ensamble Classification'
    ensambleStat.name   = 'Ensamble Statistics'

    outputs         = pd.read_csv(f'../Data/Cut/dataset1/Y/Test_{dataName}.csv', sep=";")['OutPut_class |T+1|']

    datas = pd.concat([regressions, classifications, statistics, ensambleReg, ensambleClass, ensambleStat, buying, buying2, buyingSVR, buying2SVR], axis=1)
    #datas = pd.concat([regressions, classifications, statistics], axis=1)

    # printando o shape de cada dataframe importado
    # print("regressions.shape:    ", regressions.shape)
    # print("classifications.shape:", classifications.shape)
    # print("statistics.shape:     ", statistics.shape)
    # print("ensambleReg.shape:    ", ensambleReg.shape)
    # print("ensambleClass.shape:  ", ensambleClass.shape)
    # print("ensambleStat.shape:   ", ensambleStat.shape)
    # print("buying.shape:         ", buying.shape)
    # print("buying2.shape:        ", buying2.shape)
    # print("outputs.shape:        ", outputs.shape)
    # print("buyingSVR.shape:      ", buyingSVR.shape)
    # print("buying2SVR.shape:     ", buying2SVR.shape)
    # print("datas.shape:          ", datas.shape)



    logs = pd.DataFrame(columns=['model', 'accuracy', 'f1', 'truePositives', 'trueNegatives', 'falsePositives', 'falseNegatives'])
    for column in datas.columns:
        accuracy = accuracy_score(outputs.ravel(), datas[column].ravel())
        f1 = f1_score(outputs.ravel(), datas[column].ravel())
        matrix = confusion_matrix(outputs.ravel(), datas[column].ravel())
        logs = pd.concat([logs, pd.DataFrame({'model': [column], 'accuracy': [accuracy], 'f1': [f1], 'truePositives': [matrix[0][0]], 'trueNegatives': [matrix[1][1]], 'falsePositives': [matrix[0][1]], 'falseNegatives': [matrix[1][0]]})], axis=0)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # print(logs.to_string(index=False))
    logs.to_csv(f'../Results/test/logs/class/{dataName}_logs.csv', sep=';', index=False)