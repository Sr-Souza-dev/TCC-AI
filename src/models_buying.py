from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.models import Sequential
from kerastuner.tuners import RandomSearch
from keras.layers import Dense
import pandas as pd
import keras
import numpy as np

input_dim = 6
output_dim = 2
max_trials = 50
dataName = 'dataset1'
scoring = 'neg_mean_absolute_error'


Mult_grid = {
    'estimator__kernel': ['linear', 'poly'],
    'estimator__C': np.arange(0.01, 40, 15),
    'estimator__gamma': ['scale'],
    'estimator__degree': [1, 2, 3],
    'estimator__coef0': np.arange(1, 4, 2),
}

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=6, input_dim=input_dim, activation='relu'))
    model.add(Dense(units=hp.Int('units_1', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(units=hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(units=hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(units=hp.Int('units_3', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    # Compile o modelo
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',  # Ajuste para sua tarefa
                  metrics=['accuracy'])  # Ajuste para sua tarefa
    return model

def GetModelPrediction(dN, sizeTrain):
    global input_dim, output_dim, dataName, Mult_grid, scoring
    dataName = dN

    ensamble_classification = pd.read_csv(f'../Results/train/classification/{dataName}_ensamble.csv', sep=';')
    ensamble_regression     = pd.read_csv(f'../Results/train/regression/{dataName}_ensamble.csv', sep=';')
    ensamble_statistic      = pd.read_csv(f'../Results/train/statistic/{dataName}_ensamble.csv', sep=';')
    outputs                 = pd.read_csv(f'../Data/Cut/dataset1/Y/Train_{sizeTrain}{dataName}.csv', sep=";")['OutPut_class |T+1|']

    ensamble_classification.drop('class', axis=1, inplace=True)
    ensamble_regression.drop('class', axis=1, inplace=True)
    ensamble_statistic.drop('class', axis=1, inplace=True)
    outputs.name = 'up'

    input_data = pd.concat([ensamble_classification, ensamble_regression, ensamble_statistic], axis=1)
    output_data = pd.concat([outputs, pd.Series(name = 'down', data = (outputs==0).astype(int))], axis=1)

    X_train, X_validation, Y_train, Y_validation = train_test_split(input_data, output_data, test_size=0.15, random_state=None, shuffle=False)

    # print("Ensamble_Classification.shape: ", ensamble_classification.shape)
    # print("Ensamble_Regression.shape:     ", ensamble_regression.shape)
    # print("Ensamble_Statistic.shape:      ", ensamble_statistic.shape)
    # print("Outputs.shape:                 ", outputs.shape)
    # print("Input_data.shape:              ", input_data.shape)
    # print("Output_data.shape:             ", output_data.shape)

    input_dim = input_data.shape[1]
    output_dim = output_data.shape[1]

    # Gerando modelo que utiliza o ensamble
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',  # Métrica a ser otimizada
        max_trials=max_trials,  # Número de configurações de hiperparâmetros a serem testadas
        directory=f'optmz/model{dataName}',  # Diretório para salvar resultados
        project_name='buying'  # Nome do projeto
    )

    tuner.search(X_train, Y_train, epochs=100, validation_data=(X_validation, Y_validation), verbose=0)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, Y_train, epochs=70, validation_data=(X_validation, Y_validation))

    ensamble_classification_test = pd.read_csv(f'../Results/test/classification/{dataName}_ensamble.csv', sep=';')
    ensamble_regression_test     = pd.read_csv(f'../Results/test/regression/{dataName}_ensamble.csv', sep=';')
    ensamble_statistic_test      = pd.read_csv(f'../Results/test/statistic/{dataName}_ensamble.csv', sep=';')
    
    ensamble_classification_test.drop('class', axis=1, inplace=True)
    ensamble_regression_test.drop('class', axis=1, inplace=True)
    ensamble_statistic_test.drop('class', axis=1, inplace=True)
    input_data_test = pd.concat([ensamble_classification_test, ensamble_regression_test, ensamble_statistic_test], axis=1)

    predict = model.predict(input_data_test)
    predict = pd.DataFrame(predict, columns=['up', 'down'])
    predict = pd.concat([predict['up'], predict['down'], pd.Series(name='class', data=(predict['up'] > predict['down']).astype(int))], axis=1)
    predict.to_csv(f'../Results/test/ensamble1/{dataName}.csv', sep=';', index=False)

    # ----------------------------- Otimizando SVR ----------------------------------
    print(f'                 * {dataName} - SVR ')
    Mult_grid_search = GridSearchCV(estimator = MultiOutputClassifier(SVC()), param_grid = Mult_grid, cv = 3, n_jobs = -1, verbose = 1, scoring = scoring, error_score='raise')
    Mult_grid_search.fit(input_data, output_data)
    bestSVR = Mult_grid_search.best_estimator_

    predict = bestSVR.predict(input_data_test)
    predict = pd.DataFrame(predict, columns=['up', 'down'])
    predict = pd.concat([predict['up'], predict['down'], pd.Series(name='class', data=(predict['up'] > predict['down']).astype(int))], axis=1)
    predict.to_csv(f'../Results/test/ensamble1/{dataName}_SVR.csv', sep=';', index=False)


    # **************** Gerando modelos que recebe diretamente a previsão de cada modelo *********************
    regressions     = pd.read_csv(f'../Results/train/regression/{dataName}_predictions_class.csv', sep=';')
    classifications = pd.read_csv(f'../Results/train/classification/{dataName}_predictions.csv', sep=';')
    statistics      = pd.read_csv(f'../Results/train/statistic/{dataName}_predictions_class.csv', sep=';')
    datas           = pd.concat([regressions, classifications, statistics], axis=1)
    
    input_dim  = datas.shape[1]
    output_dim = output_data.shape[1]

    X_train, X_validation, Y_train, Y_validation = train_test_split(datas, output_data, test_size=0.15, random_state=None, shuffle=False)

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',  # Métrica a ser otimizada
        max_trials=max_trials,  # Número de configurações de hiperparâmetros a serem testadas
        directory=f'optmz/model{dataName}',  # Diretório para salvar resultados
        project_name='buying2'  # Nome do projeto
    )

    tuner.search(X_train, Y_train, epochs=70, validation_data=(X_validation, Y_validation), verbose=0)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, Y_train, epochs=70, validation_data=(X_validation, Y_validation), verbose=0)

    regressions     = pd.read_csv(f'../Results/test/regression/{dataName}_predictions_class.csv', sep=';')
    classifications = pd.read_csv(f'../Results/test/classification/{dataName}_predictions.csv', sep=';')
    statistics      = pd.read_csv(f'../Results/test/statistic/{dataName}_predictions_class.csv', sep=';')
    datas_out       = pd.concat([regressions, classifications, statistics], axis=1)

    predict = model.predict(datas)
    predict = pd.DataFrame(predict, columns=['up', 'down'])
    predict = pd.concat([predict['up'], predict['down'], pd.Series(name='class', data=(predict['up'] > predict['down']).astype(int))], axis=1)
    predict.to_csv(f'../Results/test/ensamble2/{dataName}.csv', sep=';', index=False)

    # ----------------------------- Otimizando SVR ----------------------------------
    print(f'                 * {dataName} - SVR ')
    
    Mult_grid_search = GridSearchCV(estimator = MultiOutputClassifier(SVC()), param_grid = Mult_grid, cv = 3, n_jobs = -1, verbose = 1, scoring = scoring, error_score='raise')
    Mult_grid_search.fit(datas, output_data)
    bestSVR = Mult_grid_search.best_estimator_

    predict = bestSVR.predict(datas_out)
    predict = pd.DataFrame(predict, columns=['up', 'down'])
    predict = pd.concat([predict['up'], predict['down'], pd.Series(name='class', data=(predict['up'] > predict['down']).astype(int))], axis=1)
    predict.to_csv(f'../Results/test/ensamble2/{dataName}_SVR.csv', sep=';', index=False)


    
