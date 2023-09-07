import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Embedding, SimpleRNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lstmLogs = pd.DataFrame(columns=['Loss', 'Optimizer', 'Epochs', 'Batch Size', 'MSE', 'MAE', 'RMSE', 'R2_Score'])
bestLSTM = {'Loss': '', 'Optimizer': '', 'Epochs': 0, 'Batch Size': 0, 'MSE': 100000, 'MAE': 100000, 'RMSE': 100000, 'R2_Score': -10000000, 'model': None}

cnnLogs = pd.DataFrame(columns=['Loss', 'Optimizer', 'Epochs', 'Batch Size', 'MSE', 'MAE', 'RMSE', 'R2_Score'])
bestCNN = {'Loss': '', 'Optimizer': '', 'Epochs': 0, 'Batch Size': 0, 'MSE': 100000, 'MAE': 100000, 'RMSE': 100000, 'R2_Score': -10000000, 'model': None}

rnnLogs = pd.DataFrame(columns=['Loss', 'Optimizer', 'Epochs', 'Batch Size', 'MSE', 'MAE', 'RMSE', 'R2_Score'])
bestRNN = {'Loss': '', 'Optimizer': '', 'Epochs': 0, 'Batch Size': 0, 'MSE': 100000, 'MAE': 100000, 'RMSE': 100000, 'R2_Score': -10000000, 'model': None}

# loss = ['mae', 'mse', 'binary_crossentropy', 'categorical_crossentropy', 'hinge', 'logcosh']
# optimizer = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']
# epochs = np.arange(10, 100, 20)
# batch_size = np.arange(10, 100, 20)



loss = ['mae']
optimizer = ['adam']
epochs = np.arange(10, 40, 20)
batch_size = np.arange(10, 40, 20)



def saveModelChanges(model, modelName, values, X_test, Y_test):
    Y_pred = model.predict(X_test, verbose=0)
    Y_pred = Y_pred.ravel()

    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)

    # print("Pred: ",Y_pred[1:10])
    # print("Real: ",Y_test[1:10])
    # print(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R2_Score: {r2}')

    if(modelName == 'LSTM'):
        lstmLogs.loc[len(lstmLogs)] = [values[0], values[1], values[2], values[3], mse, mae, rmse, r2]
        if(bestLSTM['MSE'] > mse and bestLSTM['MAE'] > mae and bestLSTM['RMSE'] > rmse and bestLSTM['R2_Score'] < r2):
            bestLSTM['Loss'] = values[0]
            bestLSTM['Optimizer'] = values[1]
            bestLSTM['Epochs'] = values[2]
            bestLSTM['Batch Size'] = values[3]
            bestLSTM['MSE'] = mse
            bestLSTM['MAE'] = mae
            bestLSTM['RMSE'] = rmse
            bestLSTM['R2_Score'] = r2
            bestLSTM['model'] = model
    elif(modelName == 'CNN'):
        cnnLogs.loc[len(cnnLogs)] = [values[0], values[1], values[2], values[3], mse, mae, rmse, r2]
        if(bestCNN['MSE'] > mse and bestCNN['MAE'] > mae and bestCNN['RMSE'] > rmse and bestCNN['R2_Score'] < r2):
            bestCNN['Loss'] = values[0]
            bestCNN['Optimizer'] = values[1]
            bestCNN['Epochs'] = values[2]
            bestCNN['Batch Size'] = values[3]
            bestCNN['MSE'] = mse
            bestCNN['MAE'] = mae
            bestCNN['RMSE'] = rmse
            bestCNN['R2_Score'] = r2
            bestCNN['model'] = model
    elif(modelName == 'RNN'):
        rnnLogs.loc[len(rnnLogs)] = [values[0], values[1], values[2], values[3], mse, mae, rmse, r2]
        if(bestRNN['MSE'] > mse and bestRNN['MAE'] > mae and bestRNN['RMSE'] > rmse and bestRNN['R2_Score'] < r2):
            bestRNN['Loss'] = values[0]
            bestRNN['Optimizer'] = values[1]
            bestRNN['Epochs'] = values[2]
            bestRNN['Batch Size'] = values[3]
            bestRNN['MSE'] = mse
            bestRNN['MAE'] = mae
            bestRNN['RMSE'] = rmse
            bestRNN['R2_Score'] = r2
            bestRNN['model'] = model

def getLSTMModel(input_shape, loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def getLSTMModelOptimized(dataName, X_train, Y_train, X_test, Y_test):

    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    print('-------------------------- LSTM --------------------------')
    for l in loss:
        for o in optimizer:
            model = getLSTMModel((X_train.shape[1], X_train.shape[2]), loss=l, optimizer=o)
            for e in epochs:
                for b in batch_size:
                    model.fit(X_train, Y_train.ravel(), epochs=e, batch_size=b, verbose=0)
                    print(f'------ Loss: {l}, Optimizer: {o}, Epochs: {e}, Batch Size: {b}')
                    saveModelChanges(model, 'LSTM', [l, o, e, b], X_test, Y_test.ravel())

    lstmLogs.to_csv(f'../Results/optimization/regression/LSTM/{dataName}_Logs.csv', sep=';', index=False)
    print(bestLSTM)

def getCNNModel(input_shape, loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(64, (1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((1, 1)))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D((1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def getCNNModelOptimized(dataName, X_train, Y_train, X_test, Y_test):
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1], 1))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1], 1))

    print('-------------------------- CNN --------------------------')
    for l in loss:
        for o in optimizer:
            model = getCNNModel((X_train.shape[1], X_train.shape[2], X_train.shape[3]), loss=l, optimizer=o)
            for e in epochs:
                for b in batch_size:
                    model.fit(X_train, Y_train.ravel(), epochs=e, batch_size=b, verbose=0)
                    print(f'------ Loss: {l}, Optimizer: {o}, Epochs: {e}, Batch Size: {b}')
                    saveModelChanges(model, 'CNN', [l, o, e, b], X_test, Y_test.ravel())
    cnnLogs.to_csv(f'../Results/optimization/regression/CNN/{dataName}_Logs.csv', sep=';', index=False)
    print(bestCNN)

def getRNNModel(input_shape, loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(SimpleRNN(100, return_sequences=True, input_shape=input_shape))
    model.add(SimpleRNN(100))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def getRNNModelOptimized(dataName, X_train, Y_train, X_test, Y_test):
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    print('-------------------------- RNN --------------------------')
    for l in loss:
        for o in optimizer:
            model = getRNNModel((X_train.shape[1], X_train.shape[2]), loss=l, optimizer=o)
            for e in epochs:
                for b in batch_size:
                    model.fit(X_train, Y_train.ravel(), epochs=e, batch_size=b, verbose=0)
                    print(f'------ Loss: {l}, Optimizer: {o}, Epochs: {e}, Batch Size: {b}')
                    saveModelChanges(model, 'RNN', [l, o, e, b], X_test, Y_test.ravel())
    rnnLogs.to_csv(f'../Results/optimization/regression/RNN/{dataName}_Logs.csv', sep=';', index=False)
    print(bestRNN)

              
def GetModelsRegressionOptimized(dataName, size, test_size=0.4):
    X = pd.read_csv(f'../Data/Cut/dataset2/X/Optmz_{size}{dataName}.csv', sep=";")
    Y = pd.read_csv(f'../Data/Cut/dataset2/Y/Optmz_{size}{dataName}.csv', sep=";")
    Y = Y['OutPut |T+1|']
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size = test_size, random_state = None, shuffle = False)

    getLSTMModelOptimized(dataName, X_train, Y_train, X_test, Y_test)
    getCNNModelOptimized(dataName, X_train, Y_train, X_test, Y_test)
    getRNNModelOptimized(dataName, X_train, Y_train, X_test, Y_test)

    return bestLSTM, bestCNN, bestRNN
