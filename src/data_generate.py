import pandas as pd

# ----------------------------------------------------- CONFIGURAÇÕES -------------------------------------------------------
# Disponíveis = ['SMA','EMA','MACD','CCI', 'ADX','MTM','ROC','TSI','K','D','R']
methods = ['MTC', 'SMA','EMA','MACD','CCI', 'ADX','ROC','TSI','K','D','R']

# MTC - Moving time considered (quantos dias do passado serão utlizados nos dados de entrada)
daysQtd = 0
if ('MTC' in methods):
    daysQtd = 7
m = 4-daysQtd
cols = [m, m+1, m+2, m+3];

# SMA - Simple Moving Average
colSMA     = cols                             # Colunas de 'data' que serão utilizados para o calculo
windowSMA  = [3, 5, 15, 30, 200]              # A operação será realizada para cada granularização em cada coluna

# EMA - Exponential Moving Average
colEMA     = cols                             # Colunas de 'data' que serão utilizados para o calculo
windowEMA  = [5, 7, 9, 12, 26]                # A operação será realizada para cada granularização em cada coluna

# MACD - Moving Average Convergence/Divergence
colMACD    = cols                             # Colunas de 'data' que serão utilizados para o calculo
meanFast   = [ 4,  8, 12]                     # Valor da granulização da média móvel rápida
meanSlow   = [22, 17, 26]                     # Valor da granulização da média móvel lenta

# CCI - Commodity Channel Index
colCCI     = [m+1, m+2, m+3];                 # Tem normalmente como entrada os valores de HLC
windowCCI  = [14, 17, 18, 20]                 # valor da granulização do modelo

# ADX - Average Directional Index
colADX     = [m+1, m+2, m+3];                 # Tem como entrada os valores de HL (nessa mesma ordem)
windowADX  = [7, 14]                          # Fator de amortecimento considerado para as EMA (normalmente 14)

# MTM - Momentum indicator
colMTM     = cols                             # Colunas de 'data' que serão utilizados para o calculo
windowMTM  = [14,13]                          # A operação será realizada para cada granularização em cada coluna

# ROC - Price Rate of Change
colROC     = cols                             # Colunas de 'data' que serão utilizados para o calculo
windowROC  = [5, 10, 11, 12]                  # A operação será realizada para cada granularização em cada coluna

# TSI - True Strength Index
colTSI        = cols                          # Colunas de 'data' que serão utilizados para o calculo
fastWindTSI   = [13]                          # Janela para média exponencial rápida
slowWindTSI   = [25]                          # Janela para média exponencial lenta

# %K - Stochastic Oscillator
colK          = [m+1, m+2, m+3];              # Inserir dados de Máximo, Mínimo e Fechamento (HLC)
windowK       = [8, 10, 14]                   # Espaçamento entre as amostras (normalmente 14)

# %D - Stochastic Oscilator Average
colD          = [m+1, m+2, m+3];              # Inserir dados de Máximo, Mínimo e Fechamento (HLC)
windowD       = [14]                          # Espaçamento entre as amostras (normalmente 14) para o %K
windowDP      = [3]                           # Espaçamento entre as amostras para média simples

# %R - Williams
colR          = [m+1, m+2, m+3];              # Inserir dados de Máximo, Mínimo e Fechamento (HLC)
windowR       = [5, 14, 21]                   # Espaçamento entre as amostras (normalmente 14)
# -----------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------- FUNÇÕES ---------------------------------------------------------------
# ************************************* MTC - Moving time considered **************************************
def mtc(datas, days):
    rowq, colq = datas.shape
    
    colsName = [f'{datas.iloc[:,c].name}|T-{days}|' for c in range(colq)]
    res = pd.DataFrame(datas.values, columns = colsName)
    
    for q in range(days-1,-1,-1):
        colsName = [f'{datas.iloc[:,c].name}|T-{q}|' if q>0 else f'{datas.iloc[:,c].name}'
                    for c in range(colq)]
        newData = pd.DataFrame(datas.iloc[days-q:,:].values, columns = colsName)
        
        res = pd.concat([res, newData], axis=1)
    return res

# ************************************** SMA - Simple Moving Average **************************************
def sma(datas, windows):
    rowsq, colsq = datas.shape
    res = pd.DataFrame({})
    for c in range(colsq):
        colsName = [f'{datas.iloc[:,c].name}_SMA_{w}' for w in windows]
        newCols  = pd.DataFrame({}, columns = colsName) 
        cont = 0
        for w in windows:
            newCols[colsName[cont]] = datas.iloc[:,c].rolling(window=w).mean()
            cont += 1
        
        res = pd.concat([res, newCols], axis=1)  
    return res

# ************************************ EMA - Exponetial Moving Average ***********************************
def ema(datas, windows):
    rowsq, colsq = datas.shape
    res = pd.DataFrame({})
    for c in range(colsq):
        colsName = [f'{datas.iloc[:,c].name}_EMA_{w}' for w in windows]
        newCols  = pd.DataFrame({}, columns = colsName) 
        cont = 0
        for w in windows:
            newCols[colsName[cont]] = datas.iloc[:,c].ewm(span=w).mean()
            cont += 1
        
        res = pd.concat([res, newCols], axis=1)  
    return res

# ******************************* MACD - Moving Average Convergence/Divergence ***************************
def macd(datas, fast, slow):
    rowsq, colsq = datas.shape
    res = pd.DataFrame({})
    for c in range(colsq):
        colsName = [f'{datas.iloc[:,c].name}_MACD_{fast[i]}:{slow[i]}' for i in range(len(fast))]
        newCols  = pd.DataFrame({}, columns = colsName) 
        
        for i in range(len(fast)):
            newCols[colsName[i]] = (datas.iloc[:,c].ewm(span=fast[i]).mean() 
                                    - datas.iloc[:,c].ewm(span=slow[i]).mean())
        
        res = pd.concat([res, newCols], axis=1)  
    return res

# ************************************ CCI - Commodity Channel Index **************************************
def cci(datas, window):
    TP = datas.mean(axis = 1)
    res = pd.DataFrame({})
    for w in window:
        MA = TP.rolling(window=w).mean()
        DP = TP.rolling(window=w).std()
        newCols  = pd.DataFrame({f'HLC_CCI_{w}': (TP-MA)/(0.015*DP)}) 
        res = pd.concat([res, newCols], axis=1)  
    return res


# ************************************ ADX - Average Directional Index ***********************************
def adx(datas, window):    
    TR   = pd.Series([max(datas.iloc[i,0]-datas.iloc[i,1], abs(datas.iloc[i,0]-datas.iloc[i-1,2]), 
                          abs(datas.iloc[i,1]-datas.iloc[i-1,2])) for i in range(len(datas.iloc[:,0]))])
    TR[0] = TR[1]
    res = pd.DataFrame({})
    for w in window:
        ATR   = TR.ewm(span=w).mean()
        items = ['Nan'] 
        for i in range(len(ATR)-1):
            items.append((ATR[i]*(w-1)+TR[i+1])/(w))     
        newCols = pd.DataFrame({f'ADX_{w}':items})
        res = pd.concat([res, newCols], axis=1) 
    return res

# ************************************** MTM - Momentum indicator ****************************************
def mtm(datas, window):
    rowq, colq = datas.shape  
    res = pd.DataFrame({})
    for c in range(colq):
        for w in window:
            items = []
            for r in range(rowq):
                items.append(datas.iloc[r,c] - datas.iloc[r-w,c] if r-w > 0 else 'NaN') 
            newCols = pd.DataFrame({f'{datas.iloc[:,c].name}_MTM_{w}':items})
            res = pd.concat([res, newCols], axis=1) 
    return res

# ************************************ ROC - Price Rate of Change ****************************************
def roc(datas, window):
    rowq, colq = datas.shape  
    res = pd.DataFrame({})
    for c in range(colq):
        for w in window:
            items = []
            for r in range(rowq):
                items.append(100*((datas.iloc[r,c]-datas.iloc[r-w,c])/datas.iloc[r-w,c]) if r-w > 0 else 'NaN') 
            newCols = pd.DataFrame({f'{datas.iloc[:,c].name}_ROC_{w}':items})
            res = pd.concat([res, newCols], axis=1) 
    return res

# ************************************ TSI - True Strength Index *****************************************
def tsi(datas, fast, slow):
    rowq, colq = datas.shape
    res = pd.DataFrame({})
    for c in range(colq):
        PC  = [datas.iloc[l-1,c]-datas.iloc[l,c] if l>=1 else 'NaN' for l in range(rowq)]
        PCM = [abs(PC[i]) if i>=1 else 'NaN' for i in range(rowq)]
        for w in  range(len(fast)):
            num = (pd.Series(PC).ewm(span=fast[w]).mean()).ewm(span=slow[w]).mean()
            den = (pd.Series(PCM).ewm(span=fast[w]).mean()).ewm(span=slow[w]).mean()
            newCols = pd.DataFrame({f'{datas.iloc[:,c].name}_TSI_{fast[w]}:{slow[w]}':((num/den)*100)})  
            res = pd.concat([res, newCols], axis=1)
    return res

# ************************************** %K - Stochastic Oscillator ***************************************
def k(datas, window):
    rowq = len(datas.iloc[:,0])
    res = pd.DataFrame({})

    for w in window:
        items = [] 
        for i in range(rowq):
            denominador = datas.iloc[i-w,0]-datas.iloc[i-w,1]
            
            if denominador < 0.01:
                denominador = 0.01
                
            if i-w >= 0 and i-w < rowq:
                items.append((datas.iloc[i,2]-datas.iloc[i-w,1])/denominador)
            else:
                items.append('NaN')  # Valor padrão caso ocorra divisão por zero
            
        newCols = pd.DataFrame({f'K%_{w}': items})
        res = pd.concat([res, newCols], axis=1)
    
    return res

          
# *********************************** %D - Stochastic Oscilator Average ***********************************
def d(datas, windowK, windowMean):
    datas = k(datas, windowK)
    datas = sma(datas,windowMean)
    return datas

# ******************************************** %R - Williams **********************************************
def r(datas, window):
    rowq = len(datas.iloc[:, 0])
    res = pd.DataFrame({})
      
    for w in window:
        items = [] 
        for i in range(rowq):
            denominador = datas.iloc[i - w, 0] - datas.iloc[i - w, 1]
            if denominador < 0.01:
                denominador = 0.01  # Define o denominador como 0.01 se for igual a zero
            
            numerator = datas.iloc[i - w, 0] - datas.iloc[i, 2]
            ratio = numerator / denominador if i - w >= 0 and i - w < rowq else 'NaN'
            items.append(ratio)
                
        newCols = pd.DataFrame({f'R%_{w}': items})
        res = pd.concat([newCols, res], axis=1)
    
    return res



# ------------------------------------------- Generate -------------------------------------------------------
def calculate(data, methods):
    datas  = []
    bounds = []
    if('SMA' in methods):
        datas.append(sma(data.iloc[:,colSMA], windowSMA))
        bounds.append(max(windowSMA))
    if('EMA' in methods):
        datas.append(ema(data.iloc[:,colEMA], windowEMA))
        bounds.append(max(windowEMA))
    if('MACD' in methods):
        datas.append(macd(data.iloc[:,colMACD], meanFast, meanSlow))
        bounds.append(max(max(meanFast), max(meanSlow)))
    if('CCI' in methods):
        datas.append(cci(data.iloc[:,colCCI], windowCCI))
        bounds.append(max(windowCCI))
    if('ADX' in methods):
        datas.append(adx(data.iloc[:,colADX], windowADX))
        bounds.append(max(windowADX))
    if('MTM' in methods):
        datas.append(mtm(data.iloc[:,colMTM], windowMTM))
        bounds.append(max(windowMTM))
    if('ROC' in methods):
        datas.append(roc(data.iloc[:,colROC], windowROC))
        bounds.append(max(windowROC))
    if('TSI' in methods):
        datas.append(tsi(data.iloc[:,colTSI], fastWindTSI, slowWindTSI))
        bounds.append(max(max(fastWindTSI), max(slowWindTSI)))
    if('K' in methods):
        datas.append(k(data.iloc[:,colK], windowK))
        bounds.append(max(windowK))
    if('D' in methods):
        datas.append(d(data.iloc[:,colD], windowD, windowDP))
        bounds.append(max(max(windowD), max(windowDP)))
    if('R' in methods):
        datas.append(r(data.iloc[:,colR], windowR))
        bounds.append(max(windowR))
    return datas, bounds
        

def importExternalData(inputDataName, outputName):
    data = pd.read_csv(f'../Data/Collected/{inputDataName}.csv', sep=";", encoding='latin1') 

    # Realiza a filtragem na base de dados
    data = data.iloc[:,3:7]

    # Convert a base de dados de string para float
    def converter_para_float(valor):
        if(type(valor) == str):
            value = valor.replace(".", "")
            return float(value.replace(",", "."))
        return valor
    
    data = data.applymap(converter_para_float)

    # Define qual é a saída desejada
    rowq, _ = data.shape

    dataOut = pd.DataFrame({'OutPut |T+1|' : (data.loc[1:, outputName])})
    data = data.iloc[:rowq-1, :]

    # Verifica se tem espaçamento de dados passados
    if('MTC' in methods):
        data = mtc(data, daysQtd)
        dataOut = pd.DataFrame({'OutPut |T+1|' : (data.loc[1:rowq-daysQtd-2, outputName])})
        data = data.iloc[:rowq-daysQtd-2, :]

    return data, dataOut

# -----------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------- FUNÇÃO PRINCIPAL ---------------------------------------------------------------
def Generate(inputDataName, outputName):
    data, dataOut = importExternalData(inputDataName, outputName)
    datas, bounds = calculate(data, methods)
    datas.append(data)
    bounds.append(0)

    classve = [0 if dataOut.values[i][0] < dataOut.values[i-1][0] else 1 for i in range(1, dataOut.shape[0])]
    classve.insert(0,0)
    classfi = pd.DataFrame({'OutPut_class |T+1|' : classve})

    dataOut.reset_index(drop=True, inplace=True)
    dataOut = pd.concat([dataOut, classfi], axis=1)  

    # filtra as linhas invalidas
    data = pd.concat(datas, axis=1)
    data = data.iloc[(max(bounds)+1): ,:]
    dataOut = dataOut.iloc[(max(bounds)+1):,:]

    # Salva os dados
    data.to_csv(f'../Data/Generated/{inputDataName}_IN.csv', index=False, sep = ';')  
    dataOut.to_csv(f'../Data/Generated/{inputDataName}_OUT.csv', index=False, sep = ';') 
    pd.DataFrame(data.loc[:,outputName]).to_csv(f'../Data/Selected/statistic/{inputDataName}_IN.csv', index=False, sep = ';')  

    return data.shape, dataOut.shape
# -----------------------------------------------------------------------------------------------------------------------------------
