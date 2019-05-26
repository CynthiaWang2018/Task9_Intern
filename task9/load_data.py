import pandas as pd
import numpy as np

def load_data(x_path='data/task9_train_x.csv', y_path='data/task9_train_y.csv'):
    train_x = pd.read_csv(x_path, header=None)
    train_y = pd.read_csv(y_path, header=None)

    np_train_x = train_x.values # [1000, 10]
    # np_train_x[1] [ 1.24925374 -1.3495958   4.13685968  4.5149551   3.75611164  1.80277989
    #       4.48090841 -3.65859769  4.87159557 -0.69982146]
    np_train_y = train_y.values # [1000, 10]

    np_train_xy = np.append(np_train_x[:,:,np.newaxis], np_train_y[:,:,np.newaxis],2)
    # 其中的2说的是在第二个维度上append  [1000,10,1]*2 -> [1000,10,2]
    # [[ 1.24925374  1.23647908]
    #  [-1.3495958   1.15841264]
    #  [ 4.13685968 -2.27545424]
    #  [ 4.5149551  -0.69555445]
    #  [ 3.75611164 -3.54115764]
    #  [ 1.80277989 -1.20486861]
    #  [ 4.48090841 -0.84536699]
    #  [-3.65859769 -3.97096198]
    #  [ 4.87159557  0.888881  ]
    #  [-0.69982146  3.53269976]]
    return np_train_x, np_train_y, np_train_xy

def load_test_data(x_path='data/task9_evaluate_finetune_x.csv', y_path='data/task9_evaluate_finetune_y.csv'):
    train_x = pd.read_csv(x_path, header=None)
    train_y = pd.read_csv(y_path, header=None)

    np_train_x = train_x.values  # [100, 5]
    np_train_y = train_y.values  # [100, 5]
    np_train_xy = np.append(np_train_x[:, :, np.newaxis], np_train_y[:, :, np.newaxis], 2)

    return np_train_x, np_train_y, np_train_xy

def load_result_data(x_path='data/task9_evaluate_x.csv'):
    train_x = pd.read_csv(x_path, header=None)

    np_train_x = train_x.values  # [100, 5]

    return np_train_x