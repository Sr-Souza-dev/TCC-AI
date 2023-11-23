import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from kerastuner.tuners import RandomSearch
from keras import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, SimpleRNN, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

optz_size = 50
data_name = 'dataset1'

input_shape_lstm = (1, 4)
input_shape_mlp  = 4
input_shape_rnn  = (4, 1)

# Função para criar o modelo LSTM
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('unitsLSTM_1', min_value=20, max_value=100, step=20),
                   input_shape=input_shape_lstm,
                   return_sequences=True,
                   activation=hp.Choice('lstm_activation_1', ['relu', 'tanh', 'sigmoid'])))
    model.add(LSTM(units=hp.Int('unitsLSTM_2', min_value=20, max_value=100, step=20),
                   return_sequences=True,
                   activation=hp.Choice('lstm_activation_2', ['relu', 'tanh', 'sigmoid'])))
    model.add(LSTM(units=hp.Int('unitsLSTM_3', min_value=20, max_value=100, step=20),
                   return_sequences=False,  # A última camada pode ter return_sequences=False
                   activation=hp.Choice('lstm_activation_3', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dense(1))
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'Adamax', 'SGD', 'Ftrl', 'RMSprop']), loss='mae')
    return model




def build_mlp_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('unitsMLP_1', min_value=12, max_value=64, step=16),
                    input_dim=input_shape_mlp,
                    activation=hp.Choice('dense_activation_1', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dense(units=hp.Int('unitsMLP_2', min_value=12, max_value=64, step=16),
                    activation=hp.Choice('dense_activation_2', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dense(units=hp.Int('unitsMLP_3', min_value=12, max_value=64, step=16),
                    activation=hp.Choice('dense_activation_3', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dense(units=hp.Int('unitsMLP_4', min_value=12, max_value=64, step=16),
                    activation=hp.Choice('dense_activation_4', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'Adamax', 'SGD', 'Ftrl', 'RMSprop']), loss='mae')
    return model



def build_rnn_model(hp):
    model = Sequential()
    model.add(SimpleRNN(units=hp.Int('rnn_units_1', min_value=32, max_value=128, step=16),
                        return_sequences=True,
                        input_shape=input_shape_rnn,
                        activation=hp.Choice('rnn_activation_1', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=hp.Int('rnn_units_2', min_value=32, max_value=128, step=16),
                        return_sequences=True,
                        activation=hp.Choice('rnn_activation_2', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=hp.Int('rnn_units_3', min_value=32, max_value=128, step=16),
                        return_sequences=True,
                        activation=hp.Choice('rnn_activation_3', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=hp.Int('rnn_units_4', min_value=32, max_value=128, step=16),
                        return_sequences=True,
                        activation=hp.Choice('rnn_activation_4', ['relu', 'tanh', 'sigmoid'])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'Adamax', 'SGD', 'Ftrl', 'RMSprop']), loss='mae')
    return model

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
              
def GetModelsRegressionOptimized(dataName, sizeTrain):
    global input_shape_lstm, input_shape_mlp, input_shape_rnn, lstmTuner, mlpTuner, rnnTuner, data_name
    data_name = dataName

    X = pd.read_csv(f'../Data/Cut/dataset2/X/Train_{sizeTrain}{dataName}.csv', sep=";")
    Y = pd.read_csv(f'../Data/Cut/dataset2/Y/Train_{sizeTrain}{dataName}.csv', sep=";")['OutPut |T+1|']

    X = X.values.astype('float32')
    Y = Y.values.astype('float32')

    X_train, X_validation, Y_train, Y_validation = ms.train_test_split(X, Y, test_size = 0.15, random_state = None, shuffle = False)

    # Y_train              = Y_train.ravel()
    # Y_validation         = Y_validation.ravel()

    shape = X_train.shape
    print(f"{dataName} Shape: ", shape)

    bestLSTM = None
    bestMLP  = None
    bestRNN  = None

    #----------------------------------- Otimiza o LSTM -------------------------------------------
    try:
        X_train_reshape      = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_validation_reshape = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))
        input_shape_lstm = (1, 4)
        lstmTuner = RandomSearch(
            build_lstm_model,
            objective='val_loss',
            max_trials=optz_size,  # Escolha o número desejado de tentativas
            directory=f'optmz/model{data_name}/regression',  # Diretório para salvar os resultados
            project_name='lstm')

        lstmTuner.search(x=X_train_reshape, y=Y_train, epochs=100, validation_data=(X_validation_reshape, Y_validation), verbose=0)

        print(f"                     * {dataName} - LSTM ")
        bestLSTM = lstmTuner.get_best_models(num_models=1)[0]
        bestLSTM.save(f'../Results/optimization/regression/LSTM/{dataName}_model.h5')
        print("----------------------- LSTM Summary: -----------------------")
        bestLSTM.summary()
    except:
        print(f"[Erro] ao otimizar LSTM - {dataName}")
    
    # ------------------------------------ Otimiza o MLP -------------------------------------------
    try:
        input_shape_mlp  = 4
        mlpTuner = RandomSearch(
            build_mlp_model,
            objective='val_loss',
            max_trials=optz_size,  # Escolha o número desejado de tentativas
            directory=f'optmz/model{data_name}/regression',  # Diretório para salvar os resultados
            project_name='mlp')
        mlpTuner.search(x=X_train, y=Y_train, epochs=100, validation_data=(X_validation, Y_validation), verbose=0)
        print(f"                     * {dataName} - MLP ")
        bestMLP  = mlpTuner.get_best_models(num_models=1)[0]
        bestMLP.save(f'../Results/optimization/regression/MLP/{dataName}_model.h5')
        print("----------------------- MLP Summary: -----------------------")
        bestMLP.summary()
    except:
        print(f"[Erro] ao otimizar MLP - {dataName}")

    # ----------------------------------- Otimiza o RNN -------------------------------------------
    try:
        X_train_reshape      = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_validation_reshape = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

        input_shape_rnn = (1, 4)
        X_train_reshape = np.swapaxes(X_train_reshape, 1, 2)
        X_validation_reshape = np.swapaxes(X_validation_reshape, 1, 2)

        rnnTuner = RandomSearch(
            build_rnn_model,
            objective='val_loss',
            max_trials=optz_size,  # Escolha o número desejado de tentativas
            directory=f'optmz/model{data_name}/regression',  # Diretório para salvar os resultados
            project_name='rnn')
        rnnTuner.search(x=X_train_reshape, y=Y_train, epochs=100, validation_data=(X_validation_reshape, Y_validation), verbose=0)
        print(f"                     * {dataName} - RNN ")
        bestRNN  = rnnTuner.get_best_models(num_models=1)[0]
        bestRNN.save(f'../Results/optimization/regression/RNN/{dataName}_model.h5')
        print("----------------------- RNN Summary: -----------------------")
        bestRNN.summary()
    except:
        print(f"[Erro] ao otimizar RNN - {dataName}")

    return bestLSTM, bestMLP, bestRNN

def GetModelsRegression(dataName):
    LSTM = load_model(f'../Results/optimization/regression/LSTM/{dataName}_model.h5')
    MLP = load_model(f'../Results/optimization/regression/MLP/{dataName}_model.h5')
    RNN = load_model(f'../Results/optimization/regression/RNN/{dataName}_model.h5')
    return LSTM, MLP, RNN

def GetRegressionPredictions(dataName, Names, Models, X_test, Y_test, X_train, Y_train):
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Calcula o resultado no conjunto de teste
    results = pd.DataFrame()
    results_Class = pd.DataFrame()

    for name, model in zip(Names, Models):
        series = pd.Series(name=name, data=(model.predict(X_test)).ravel())
        serie_last = Y_test.shift(1)
        series_class = pd.Series(name=name, data = (series > serie_last).astype(int))
        results_Class = pd.concat([results_Class, series_class], axis=1)
        results = pd.concat([results, series], axis=1)

    # Calcula o resultado no conjunto de treinameito
    results_ClassTrain = pd.DataFrame()
    for name, model in zip(Names, Models):
        series = pd.Series(name=name, data=(model.predict(X_train)).ravel())
        serie_last = Y_train.shift(1)
        series_class = pd.Series(name=name, data = (series > serie_last).astype(int))
        results_ClassTrain = pd.concat([results_ClassTrain, series_class], axis=1)

    results_ClassTrain.to_csv(f'../Results/train/regression/{dataName}_predictions_class.csv', sep=';', index=False)
    results_Class.to_csv(f'../Results/test/regression/{dataName}_predictions_class.csv', sep=';', index=False)
    results.to_csv(f'../Results/test/regression/{dataName}_predictions.csv', sep=';', index=False)