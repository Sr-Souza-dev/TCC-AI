import sklearn.model_selection as ms
import pandas as pd

def Cut(dataName, size):
    dataset1 = pd.read_csv(f'../Data/Selected/dataset1/{dataName}_IN_class.csv', sep = ';')  
    dataset2 = pd.read_csv(f'../Data/Selected/dataset2/{dataName}_IN_regre.csv', sep = ';')
    estatist = pd.read_csv(f'../Data/Selected/statistic/{dataName}_IN.csv', sep = ';')
    Y        = pd.read_csv(f'../Data/Selected/{dataName}_Out.csv',      sep = ';') 
    # print(f'Out shape:          {Y.shape}')

    # Split do dataset 1
    X_data, X_optmz, Y_data, Y_optmz = ms.train_test_split(dataset1, Y, test_size = size[0], random_state = None, shuffle = False)
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X_data, Y_data, test_size = size[2], random_state = None, shuffle = False)
    
    X_optmz.to_csv(f'../Data/Cut/dataset1/X/Optmz_{size[0]}{dataName}.csv', index=False, sep = ';') 
    Y_optmz.to_csv(f'../Data/Cut/dataset1/Y/Optmz_{size[0]}{dataName}.csv', index=False, sep = ';') 
    X_train.to_csv(f'../Data/Cut/dataset1/X/Train_{size[1]}{dataName}.csv', index=False, sep = ';') 
    Y_train.to_csv(f'../Data/Cut/dataset1/Y/Train_{size[1]}{dataName}.csv', index=False, sep = ';') 
    X_test.to_csv( f'../Data/Cut/dataset1/X/Test_{dataName}.csv',  index=False, sep = ';') 
    Y_test.to_csv( f'../Data/Cut/dataset1/Y/Test_{dataName}.csv',  index=False, sep = ';') 

    # print(f'-------- Dataset 1 shape: {dataset1.shape} --------')
    # print(f'    X_optmz shape:      {X_optmz.shape}')
    # print(f'    Y_optmz shape:      {Y_optmz.shape}')
    # print(f'    X_train shape:      {X_train.shape}')
    # print(f'    Y_train shape:      {Y_train.shape}')
    # print(f'    X_test shape:       {X_test.shape}')
    # print(f'    Y_test shape:       {Y_test.shape}')

    # Split do dataset 2
    X_data, X_optmz, Y_data, Y_optmz = ms.train_test_split(dataset2, Y, test_size = size[0], random_state = None, shuffle = False)
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X_data, Y_data, test_size = size[2], random_state = None, shuffle = False)
    
    X_optmz.to_csv(f'../Data/Cut/dataset2/X/Optmz_{size[0]}{dataName}.csv', index=False, sep = ';') 
    Y_optmz.to_csv(f'../Data/Cut/dataset2/Y/Optmz_{size[0]}{dataName}.csv', index=False, sep = ';') 
    X_train.to_csv(f'../Data/Cut/dataset2/X/Train_{size[1]}{dataName}.csv', index=False, sep = ';') 
    Y_train.to_csv(f'../Data/Cut/dataset2/Y/Train_{size[1]}{dataName}.csv', index=False, sep = ';') 
    X_test.to_csv( f'../Data/Cut/dataset2/X/Test_{dataName}.csv',  index=False, sep = ';') 
    Y_test.to_csv( f'../Data/Cut/dataset2/Y/Test_{dataName}.csv',  index=False, sep = ';') 

    # print(f'-------- Dataset 2 shape: {dataset2.shape} --------')
    # print(f'    X_optmz shape:      {X_optmz.shape}')
    # print(f'    Y_optmz shape:      {Y_optmz.shape}')
    # print(f'    X_train shape:      {X_train.shape}')
    # print(f'    Y_train shape:      {Y_train.shape}')
    # print(f'    X_test shape:       {X_test.shape}')
    # print(f'    Y_test shape:       {Y_test.shape}')
    
    # Split do statistic
    X_data, X_optmz, Y_data, Y_optmz = ms.train_test_split(estatist, Y, test_size = size[0], random_state = None, shuffle = False)
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X_data, Y_data, test_size = size[2], random_state = None, shuffle = False)
    
    X_optmz.to_csv(f'../Data/Cut/statistic/X/Optmz_{size[0]}{dataName}.csv', index=False, sep = ';') 
    Y_optmz.to_csv(f'../Data/Cut/statistic/Y/Optmz_{size[0]}{dataName}.csv', index=False, sep = ';') 
    X_train.to_csv(f'../Data/Cut/statistic/X/Train_{size[1]}{dataName}.csv', index=False, sep = ';') 
    Y_train.to_csv(f'../Data/Cut/statistic/Y/Train_{size[1]}{dataName}.csv', index=False, sep = ';') 
    X_test.to_csv( f'../Data/Cut/statistic/X/Test_{dataName}.csv',  index=False, sep = ';') 
    Y_test.to_csv( f'../Data/Cut/statistic/Y/Test_{dataName}.csv',  index=False, sep = ';') 

    # print(f'-------- Estatistic shape:   {estatist.shape} --------')
    # print(f'    X_optmz shape:      {X_optmz.shape}')
    # print(f'    Y_optmz shape:      {Y_optmz.shape}')
    # print(f'    X_train shape:      {X_train.shape}')
    # print(f'    Y_train shape:      {Y_train.shape}')
    # print(f'    X_test shape:       {X_test.shape}')
    # print(f'    Y_test shape:       {Y_test.shape}')