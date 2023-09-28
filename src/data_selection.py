from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from scipy.stats import kruskal
from enum import Enum 

import pandas as pd
import numpy as np

# ---------------------------------------- Configurações ----------------------------------------- #
class methods(Enum):
    # Modelos de previsão 
    RandomForest = 1               # Classificação / Regressão
    Lasso = 2                      # Classificação / Regressão
    ElasticNet = 3                 # Classificação / Regressão
    
    # Métodos estatísticos
    FsFisher = 4                   # Classificação (class em int -> int(y))
    FsGini = 5                     # Classificação (class em int -> int(y))
    KruskalWallis = 6              # Classificação / Regressão
    Chi2 = 7                       # Classificação (class em int -> int(y))
    FRegression = 8                # Regressão
    MutualRegression = 9           # Regressão

usedMethodsD1  = [methods.FsFisher, methods.FsGini, methods.Chi2, methods.RandomForest]
usedMethodsD2  = [methods.MutualRegression, methods.Lasso, methods.ElasticNet, methods.RandomForest]

usedMethodsDB1 =  methods.KruskalWallis
usedMethodsDB2 =  methods.KruskalWallis

filterEachD1   = [20, 20, 20, 20]
filterEachD2   = [20, 20, 20, 20]
dataset1_2     = [4, 4]

# RandomForest
treesqtd = 200

#Lasso
lassoAlpha = 0.05

#ElasticNet
elasticAlpha  = 0.5
elasticRation = 0.5

#---------------------------------------- Funções ----------------------------------------- #
# Funções de seleção de variáveis

# ****************************************** RandomForest ****************************************
def randomForest(X,Y,qtdF):
    model = RandomForestRegressor(n_estimators=treesqtd, random_state=42)
    model.fit(X, Y.values.ravel())
    importances = pd.DataFrame({'feature': X.columns, 'importance':model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False)
    usedVars = importances.index[0:qtdF]
    return X.iloc[:,usedVars]

# ********************************************* Fisher *******************************************
def fisher(X, Y, qtdF):
    #X, Y = X.iloc[:10,:], Y.iloc[:10]
    colq = X.shape[1]
    maxY = int(np.max(Y)+1)
    minY = int(np.min(Y))
    out  = pd.DataFrame({'W': np.zeros(colq)})
    classIdx = [np.where(Y==j)[0] for j in range(minY, maxY)]
    classQtd = [len(classIdx[j])  for j in range(maxY-minY)]
    for i in range(colq):
        errMed  = 0
        varPon  = 0
        col     = X.iloc[:,i]
        colMean = np.mean(col)
        for j in range(maxY-minY):
            classMean = np.mean(col.iloc[classIdx[j]])
            classVari = np.var(col.iloc[classIdx[j]], ddof=1)
            errMed   += classQtd[j] * (classMean-colMean)**2
            varPon   += classVari * classQtd[j]
        if errMed == 0:
            out['W'][i] = 0
        elif varPon == 0:
            out['W'][i] = 100
        else:
            out['W'][i] = errMed/varPon       
    return X.iloc[:,(out['W'].sort_values(ascending=False).index)[:qtdF]]
      
# ********************************************* Gine ********************************************
def gini(X, Y, qtd):
    a, n = X.shape
    W = np.zeros((n, 1))

    for i in range(n):
        values = np.unique(X.iloc[:, i])
        v = len(values)
        W[i] = 0.5
        for j in range(v):
            left_Y = Y[X.iloc[:, i] <= values[j]]
            right_Y = Y[X.iloc[:, i] > values[j]]

            gini_left = 0
            gini_right = 0

            if len(left_Y) > 0:
                for k in range(np.min(Y), np.max(Y) + 1):
                    gini_left += (len(left_Y[left_Y == k]) / len(left_Y))**2
                gini_left = 1 - gini_left

            if len(right_Y) > 0:
                for k in range(np.min(Y), np.max(Y) + 1):
                    gini_right += (len(right_Y[right_Y == k]) / len(right_Y))**2
                gini_right = 1 - gini_right

            current_gini = 0
            if len(Y) > 0:
                current_gini = (len(left_Y) * gini_left + len(right_Y) * gini_right) / len(Y)

            if current_gini < W[i]:
                W[i] = current_gini
                       
    W = np.sort(W, axis=0)[::-1].flatten(), np.argsort(W, axis=0)[::-1].flatten()

    return X.iloc[:,W[1]]


# ***************************************** KruskalWallis ***************************************
def kruskalWallis(X, Y, qtd):
    n = X.shape[1]
    out = pd.DataFrame(np.zeros(n), columns=["W"])
    for i in range(n):
        out['W'].iloc[i] = - np.mean(kruskal(X.iloc[:,i], Y.iloc[:])[1])
    out = out.sort_values('W', ascending=False).index
    return X.iloc[:,out[:qtd]]

# ******************************************* Lasso *********************************************
def lasso(data, Y, qtd):
    model = Lasso(alpha=lassoAlpha, random_state=0)
    model.fit(data, Y)
    rate = pd.DataFrame(abs(model.coef_), columns=["W"])
    rate = rate.sort_values('W', ascending=False).index    
    return data.iloc[:, rate[:qtd]]

# **************************************** ElasticNet *******************************************
def elasticNet(data, Y, qtd):
    model = ElasticNet(alpha=elasticAlpha, l1_ratio=elasticRation)
    model.fit(data, Y)
    rate = pd.DataFrame(abs(model.coef_), columns=["W"])
    rate = rate.sort_values('W', ascending=False).index    
    return data.iloc[:, rate[:qtd]]

# ****************************************** Chi2 (X²) *******************************************
def ftChi2(data, Y, qtd):
    selector = pd.DataFrame(chi2(data.abs(), Y.values.ravel())[0], columns=["W"])
    selector = selector.sort_values('W', ascending=False).index
    return data.iloc[:, selector[:qtd]]

# *************************************** F Regression ******************************************
def fRegression(data, Y, qtd):
    selector = pd.DataFrame(f_regression(data, Y.values.ravel())[0], columns=["W"])
    selector = selector.sort_values('W', ascending=False).index
    return data.iloc[:, selector[:qtd]]

# *********************************** Mutual Info Regression*************************************
def mutualRegression(data, Y, qtd):
    mi = mutual_info_regression(data, Y.values.ravel())
    selector = pd.DataFrame(mi, columns=["W"])
    selector = selector.sort_values('W', ascending=False).index
    return data.iloc[:, selector[:qtd]]
    
# ************************************** seleção dos dados **************************************
def select(data, Y, qtd, option):
    if(option == methods.RandomForest):
        data = randomForest(data,Y.iloc[:,0],qtd) 
        
    elif(option == methods.FsFisher):
        data = fisher(data,Y.iloc[:,1],qtd) 
        
    elif(option == methods.FsGini):
        data = gini(data,Y.iloc[:,1],qtd) 
        
    elif(option == methods.KruskalWallis):
        data = kruskalWallis(data,Y.iloc[:,0],qtd)
        
    elif(option == methods.Lasso):
        data = lasso(data,Y.iloc[:,0],qtd)
        
    elif(option == methods.ElasticNet):
        data = elasticNet(data,Y.iloc[:,0],qtd)
        
    elif(option == methods.Chi2):
        data = ftChi2(data,Y.iloc[:,1],qtd)
        
    elif(option == methods.FRegression):
        data = fRegression(data,Y.iloc[:,0],qtd)
        
    elif(option == methods.MutualRegression):
        data = mutualRegression(data,Y.iloc[:,0],qtd)
    return data

def calculate(data, Y, opt1, qtd1, opt2, qtd2):
    dataset1 = pd.DataFrame({})
    dataset2 = pd.DataFrame({})
    
    print("Gerando dataset1...")
    for i in range(len(opt1)):
        print("        ->",opt1[i])
        dataset1 = dataset1.combine_first(select(data, Y, qtd1[i], opt1[i]))
    
    print("done.")
    print("Gerando dataset2...")
    for i in range(len(opt2)):
        print("        ->",opt2[i])
        dataset2 = dataset2.combine_first(select(data, Y, qtd2[i], opt2[i]))
    
    print("done.")
    
    dataset1 = select(dataset1, Y, dataset1_2[0], usedMethodsDB1)
    dataset2 = select(dataset2, Y, dataset1_2[1], usedMethodsDB2)
    
    
    return [dataset1, dataset2]


def Selection(inputDataName):
    # Inserção dos dados para seleção de variáveis
    X = pd.read_csv(f'../Data/Generated/{inputDataName}_IN.csv',  sep=";") 
    Y = pd.read_csv(f'../Data/Generated/{inputDataName}_OUT.csv', sep=";") 

    dataset1, dataset2 = calculate(X, Y, usedMethodsD1, filterEachD1, usedMethodsD2, filterEachD2)

    dataset1.to_csv(f'../Data/Selected/dataset1/{inputDataName}_IN_class.csv', index=False, sep = ';')
    dataset2.to_csv(f'../Data/Selected/dataset2/{inputDataName}_IN_regre.csv', index=False, sep = ';')

    Y.to_csv(f'../Data/Selected/{inputDataName}_Out.csv', index=False, sep = ';')

    return dataset1.shape, dataset2.shape, Y.shape