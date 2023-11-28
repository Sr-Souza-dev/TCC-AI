import pandas as pd
import matplotlib.pyplot as plt
import os

initialValue = 1000     # Valor inicial do investimento

def GetEconomyAnalyze(dataName, testSize):
    classifications = pd.read_csv(f'../Results/test/classification/{dataName}_predictions.csv', sep=";")
    regressions     = pd.read_csv(f'../Results/test/regression/{dataName}_predictions_class.csv', sep=";")
    statistics      = pd.read_csv(f'../Results/test/statistic/{dataName}_predictions_class.csv', sep=";")

    ensamble_classification         = pd.read_csv(f'../Results/test/classification/{dataName}_ensamble.csv', sep=";")['class']
    ensamble_regression             = pd.read_csv(f'../Results/test/regression/{dataName}_ensamble.csv', sep=";")['class']
    ensamble_statistics             = pd.read_csv(f'../Results/test/statistic/{dataName}_ensamble.csv', sep=";")['class']
    ensamble_classification.name    = "Ensamble Classification"
    ensamble_regression.name        = "Ensamble Regression"
    ensamble_statistics.name        = "Ensamble Statistics"

    buying          = pd.read_csv(f'../Results/test/ensamble1/{dataName}.csv', sep=";")['class']
    buying2         = pd.read_csv(f'../Results/test/ensamble2/{dataName}.csv', sep=";")['class']
    buying.name     = "buying"
    buying2.name    = "buying2"

    buyingSVR       = pd.read_csv(f'../Results/test/ensamble1/{dataName}_SVR.csv', sep=';')['class']
    buying2SVR      = pd.read_csv(f'../Results/test/ensamble2/{dataName}_SVR.csv', sep=';')['class']
    buyingSVR.name  = 'buying_SVR'
    buying2SVR.name = 'buying2SVR'

    datas  = pd.concat([classifications, regressions, statistics, ensamble_classification, ensamble_regression, ensamble_statistics, buying, buying2, buyingSVR, buying2SVR], axis=1)
    output = pd.read_csv(f'../Data/Cut/dataset1/Y/Test_{dataName}.csv', sep=";")['OutPut |T+1|']

    # print("Shapes dos dados importados:")
    # print("Classifications:             ", classifications.shape)
    # print("Regressions:                 ", regressions.shape)
    # print("Statistics:                  ", statistics.shape)
    # print("Ensamble Classification:     ", ensamble_classification.shape)
    # print("Ensamble Regression:         ", ensamble_regression.shape)
    # print("Ensamble Statistics:         ", ensamble_statistics.shape)
    # print("Buying:                      ", buying.shape)
    # print("Buying2:                     ", buying2.shape)
    # print("Datas:                       ", datas.shape)
    # print("Output:                      ", output.shape)

    # faz a operação de compra e venda em todos os modelos
    modelsHistory = pd.DataFrame()
    operationsHistory = pd.DataFrame(columns = ["model", "qtdBuying", "inirialValue", "finalValue", "percentual"])
    for name in datas.columns:
        qtdBuying = 0
        currentValue = initialValue
        purchasedValue = 0
        isPurchased = False
        purchasedHistory = []
        for decision, value in zip(datas[name], output):
            if decision == 1 and not isPurchased:
                qtdBuying += 1
                isPurchased = True
                purchasedValue = value
            elif decision == 0 and isPurchased:
                isPurchased = False
                currentValue = (currentValue / purchasedValue) * value
            
            if(isPurchased):
                purchasedHistory.append((currentValue / purchasedValue) * value)
            else:
                purchasedHistory.append(currentValue)
                
        if(isPurchased):
            currentValue = (currentValue / purchasedValue) * value
            purchasedHistory[len(purchasedHistory) - 1] = currentValue
        
        log = {
            "model": name,
            "qtdBuying": qtdBuying,
            "inirialValue": initialValue,
            "finalValue": currentValue,
            "percentual": ((currentValue - initialValue) / initialValue) * 100
        }
        operationsHistory = pd.concat([operationsHistory, pd.DataFrame(log, index=[0])], ignore_index=True)
        modelsHistory = pd.concat([modelsHistory, pd.Series(purchasedHistory, name=name)], axis=1)

    # calcula o buy and hold
    currentValue = initialValue
    buyAndHoldHistory = []
    for value in output:
        currentValue = (initialValue / output[0]) * value
        buyAndHoldHistory.append(currentValue)

    log = {
        "model": "Buy and Hold",
        "qtdBuying": 1,
        "inirialValue": initialValue,
        "finalValue": currentValue,
        "percentual": ((currentValue - initialValue) / initialValue) * 100
    }
    pd.DataFrame(log, index=[0]).to_csv(f'../Results/test/buyAndHold/{dataName}_logs.csv', sep=';', index=False)
    operationsHistory = pd.concat([operationsHistory, pd.DataFrame(log, index=[0])], ignore_index=True)
    modelsHistory = pd.concat([modelsHistory, pd.Series(buyAndHoldHistory, name="Buy and Hold")], axis=1)

    statisticsLogs = pd.read_csv(f'../Results/test/logs/class/{dataName}_logs.csv', sep=';')
    operationsHistory = pd.merge(operationsHistory, statisticsLogs, on='model', how='outer')
    operationsHistory.to_csv(f'../Results/test/logs/class/{dataName}_logs.csv', sep=';', index=False)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # print("Models History:              ", modelsHistory.shape)
    # print("Operations History:        \n", operationsHistory.to_string(index=False))

    by = modelsHistory["Buy and Hold"]
    by.to_csv(f'../Results/test/buyAndHold/{dataName}.csv', sep=';', index=False)
    modelsHistory = modelsHistory.drop(columns=["Buy and Hold"])

    # plota o gráfico de comparação entre os modelos
    dir_name = f"../Results/plots/{dataName}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for column in modelsHistory.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(modelsHistory[column], label=column)
        plt.plot(by, label="Buy and Hold")
        plt.legend()
        plt.title(f"{column} x Buy and Hold")
        plt.xlabel("Operação")
        plt.ylabel("Valor (R$)")
        plt.savefig(f"{dir_name}/{column}_x_buyAndHold.png")
        plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(by, label="Buy and Hold")
    for column in modelsHistory.columns:
        plt.plot(modelsHistory[column], label=column)
    plt.legend(loc='upper left')
    plt.title(f"Modelos x Buy and Hold")
    plt.xlabel("Operação")
    plt.ylabel("Valor (R$)")
    plt.savefig(f"{dir_name}/all.png")
    plt.close()




    