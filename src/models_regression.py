import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from kerastuner.tuners import RandomSearch
from keras import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, SimpleRNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

optz_size = 1
data_name = 'dataset1'

input_shape_lstm = (1, 1, 4)
input_shape_mlp  = (1, 1, 6)
input_shape_rnn  = (1, 1, 6)

# Função para criar o modelo LSTM
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=50, max_value=200, step=10),input_shape=input_shape_lstm, return_sequences=True))
    model.add(Dense(1))
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),loss='mae')
    return model

# Função para criar o modelo mlp
def build_mlp_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=16, max_value=64, step=16), input_shape=input_shape_rnn, activation='relu'))
    model.add(Dense(units=hp.Int('units_1', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(units=hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(units=hp.Int('units_3', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']), loss='mae')
    return model


def build_rnn_model(hp):
    model = Sequential()
    model.add(SimpleRNN(units=hp.Int('rnn_units', min_value=32, max_value=128, step=16), return_sequences=True, input_shape=input_shape_rnn))
    model.add(SimpleRNN(units=hp.Int('rnn_units', min_value=32, max_value=128, step=16)))
    model.add(Dense(1))
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),loss='mae')
    return model

              
def GetModelsRegressionOptimized(dataName, sizeTrain):
    global input_shape_lstm, input_shape_mlp, input_shape_rnn, lstmTuner, mlpTuner, rnnTuner, data_name
    data_name = dataName

    X = pd.read_csv(f'../Data/Cut/dataset2/X/Train_{sizeTrain}{dataName}.csv', sep=";")
    Y = pd.read_csv(f'../Data/Cut/dataset2/Y/Train_{sizeTrain}{dataName}.csv', sep=";")['OutPut |T+1|']

    X_train, X_validation, Y_train, Y_validation = ms.train_test_split(X, Y, test_size = 0.15, random_state = None, shuffle = False)

    Y_train              = Y_train.ravel()
    Y_validation         = Y_validation.ravel()
    X_train_reshape      = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_validation_reshape = X_validation.values.reshape((X_validation.shape[0], 1, X_validation.shape[1]))

    shape = X_train_reshape.shape
    print("Shape: ", shape)

    input_shape_lstm = (shape[1], shape[2])
    input_shape_rnn  = (shape[1], shape[2])
    input_shape_mlp  = (shape[1], shape[2], 1)

    lstmTuner = RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=optz_size,  # Escolha o número desejado de tentativas
        directory=f'optmz/model{data_name}/regression',  # Diretório para salvar os resultados
        project_name='lstm')

    mlpTuner = RandomSearch(
        build_mlp_model,
        objective='val_loss',
        max_trials=optz_size,  # Escolha o número desejado de tentativas
        directory=f'optmz/model{data_name}/regression',  # Diretório para salvar os resultados
        project_name='mlp')

    rnnTuner = RandomSearch(
        build_rnn_model,
        objective='val_loss',
        max_trials=optz_size,  # Escolha o número desejado de tentativas
        directory=f'optmz/model{data_name}/regression',  # Diretório para salvar os resultados
        project_name='rnn')

    lstmTuner.search(x=X_train_reshape, y=Y_train, epochs=100, validation_data=(X_validation_reshape, Y_validation))
    mlpTuner.search(x=X_train_reshape, y=Y_train, epochs=100, validation_data=(X_validation_reshape, Y_validation))
    rnnTuner.search(x=X_train_reshape, y=Y_train, epochs=100, validation_data=(X_validation_reshape, Y_validation))

    bestLSTM = lstmTuner.get_best_models(num_models=1)[0]
    bestMLP  = mlpTuner.get_best_models(num_models=1)[0]
    bestRNN  = rnnTuner.get_best_models(num_models=1)[0]

    bestLSTM.save(f'../Results/optimization/regression/LSTM/{dataName}_model.h5')
    bestMLP.save(f'../Results/optimization/regression/MLP/{dataName}_model.h5')
    bestRNN.save(f'../Results/optimization/regression/RNN/{dataName}_model.h5')

    print("LSTM: ", bestLSTM.summary())
    print("mlp:  ", bestMLP.summary())
    print("RNN:  ", bestRNN.summary())

    return bestLSTM, bestMLP, bestRNN

def GetModelsRegression(dataName):
    LSTM = load_model(f'../Results/optimization/regression/LSTM/{dataName}_model.h5')
    MLP = load_model(f'../Results/optimization/regression/MLP/{dataName}_model.h5')
    RNN = load_model(f'../Results/optimization/regression/RNN/{dataName}_model.h5')
    return LSTM, MLP, RNN

def GetRegressionPredictions(dataName, Names, Models, Epochs, Batchs, X_test, Y_test, X_train, Y_train):
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Calcula o resultado no conjunto de teste
    results = pd.DataFrame()
    results_Class = pd.DataFrame()

    for name, model, epoch, batch in zip(Names, Models, Epochs, Batchs):
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