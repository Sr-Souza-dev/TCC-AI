import os
import pandas as pd
import matplotlib.pyplot as plt

initialValue = 1000

# Estratégia proposta no TCC
def strategy1(up, down, output):
    if(abs(up - down) < 0.1):
        return 0
    if(up > down):
        return 1
    return -1

# Estratégia de Hurwicz
alpha = 0.6
def hurwicz(up, down, output):
    decision_matriz = [
        [up*output[0],  output[0], output[0]-(down*output[0])],     # Se eu comprar
        [-up*output[1], output[1], down*output[1]],                 # Se eu não fizer nada
        [-up*output[0], output[0], down*output[0]]                  # Se eu vender
    ]

    # decision_matriz = [
    #     [up, up+down, down],     # Se eu comprar
    #     [abs(up-down), up+down+abs(up-down), abs(up-down)],                 # Se eu não fizer nada
    #     [down, up+down, up]                  # Se eu vender
    # ]

    ponderation = []
    for linha in decision_matriz:
        value_p = alpha * max(linha) + (1 - alpha) * min(linha)
        ponderation.append(value_p)
    decision = ponderation.index(max(ponderation)) + 1

    if decision == 1: return 1
    if decision == 2: return 0
    if decision == 3: return -1
    return 0

strategies      = [strategy1, hurwicz]
strategiesNames = ["strat_TCC", "strat_hurwicz"]

def for_each_type_of_file(files, path, outputsPath, validation):
    folderName = path.split("/")[-1]

    logsBuying  = pd.DataFrame(columns = ["model", "qtdBuying", "inirialValue", "finalValue", "percentual"])

    for file in files:
        valid, fileName = validation(file)
        if valid:
            inputData  = pd.read_csv(path + "/" + file, sep=";")
            outputData = pd.read_csv(outputsPath + "/Test_" + fileName + ".csv", sep=";")['OutPut |T+1|']
            buyAndHold = pd.read_csv("../Results/test/buyAndHold/" + fileName + ".csv", sep=";")
            buyAndHolsLogs = pd.read_csv("../Results/test/buyAndHold/" + fileName + "_logs.csv", sep=";")

            for strategy, strategyName in zip(strategies, strategiesNames):
                qtdBuying = 0
                currentValue = initialValue
                purchasedValue = 0
                isPurchased = False
                purchasedHistory = []

                for index, row in inputData.iterrows():
                    decision = strategy(row['up'], row['down'], [outputData[index-1] if index-1>=0 else 0, outputData[index], outputData[index+1] if index+1<outputData.shape[0] else 0])

                    if(decision == 1 and not isPurchased):
                        qtdBuying += 1
                        isPurchased = True
                        purchasedValue = outputData[index]

                    elif(decision == -1 and isPurchased):
                        isPurchased = False
                        currentValue = (currentValue / purchasedValue) * outputData[index]

                    if(isPurchased):
                        purchasedHistory.append((currentValue / purchasedValue) * outputData[index])
                    else:
                        purchasedHistory.append(currentValue)

                if(isPurchased):
                    currentValue = (currentValue / purchasedValue) * outputData[outputData.shape[0] - 1]
                    purchasedHistory[len(purchasedHistory) - 1] = currentValue
                
                log = {
                    "model": strategyName+"_"+fileName,
                    "qtdBuying": qtdBuying,
                    "inirialValue": initialValue,
                    "finalValue": currentValue,
                    "percentual": ((currentValue - initialValue) / initialValue) * 100
                }
                logsBuying = pd.concat([logsBuying, pd.DataFrame(log, index=[0])], ignore_index=True)
                plt.figure(figsize=(10, 5))
                plt.plot(purchasedHistory,  label=strategyName)
                plt.plot(buyAndHold.values, label="Buy and Hold")
                plt.legend()
                plt.title(f"{strategyName} x Buy and Hold")
                plt.xlabel("Operação")
                plt.ylabel("Valor (R$)")
                plt.savefig(f"../Results/plots/{fileName}/{strategyName}_x_buyAndHold_{folderName}.png")
                plt.close()
            logsBuying = pd.concat([logsBuying, buyAndHolsLogs], ignore_index=True)
    
    logsBuying.to_csv(f'../Results/test/logs/economic/{folderName}_buying.csv', sep=';', index=False)


def GetStrategies():
    print("gerando estrategias...")
    classificationsPath = "../Results/test/classification"
    regressionsPath     = "../Results/test/regression"
    statisticsPath      = "../Results/test/statistic"
    ensamble1Path       = "../Results/test/ensamble1"
    ensamble2Path       = "../Results/test/ensamble2"
    outputsPath         = "../Data/Cut/dataset1/Y"

    classificationsfiles = os.listdir(classificationsPath)
    regressionsfiles     = os.listdir(regressionsPath)
    statisticsfiles      = os.listdir(statisticsPath)
    ensamble1files       = os.listdir(ensamble1Path)
    ensamble2files       = os.listdir(ensamble2Path)

    def ensambles_validation(file):
        fileName = ""
        if (file.endswith("ensamble.csv")):
            fileName = file.replace("_ensamble.csv", "")
            return True, fileName.strip()
        return False, fileName
    
    def model_validation(file):
        fileName = file.replace(".csv", "")
        return True, fileName.strip()
    
    files       = [classificationsfiles, regressionsfiles, statisticsfiles, ensamble1files, ensamble2files]
    paths       = [classificationsPath, regressionsPath, statisticsPath, ensamble1Path, ensamble2Path]
    validations = [ensambles_validation, ensambles_validation, ensambles_validation, model_validation, model_validation]

    for file, path, validation in zip(files, paths, validations):
        for_each_type_of_file(file, path, outputsPath, validation)
