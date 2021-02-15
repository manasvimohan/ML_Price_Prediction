'''
-----------------------------------------------------------------------
Created by Manasvi Mohan Sharma on 15/02/21 (dd/mm/yy)
Project Name: ML_Price_Prediction | File Name: custom_functions.py
IDE: PyCharm | Python Version: 3.8
-----------------------------------------------------------------------
                                       _ 
                                      (_)
 _ __ ___   __ _ _ __   __ _ _____   ___ 
| '_ ` _ \ / _` | '_ \ / _` / __\ \ / / |
| | | | | | (_| | | | | (_| \__ \\ V /| |
|_| |_| |_|\__,_|_| |_|\__,_|___/ \_/ |_|

GitHub:   https://github.com/manasvimohan
Linkedin: https://www.linkedin.com/in/manasvi-mohan-sharma-119375168/
Website:  https://www.manasvi.co.in/
-----------------------------------------------------------------------
Project Information:
Considering stock price data as a time series, regression based approach
is used to create models using Tensorflow module to predict prices of the
stock in future.

About this file:
This file contains all the custom functions
-----------------------------------------------------------------------
'''

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pandas_datareader import data as web

from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
fig_breadth = 14
fig_height = fig_breadth/2
# print(plt.style.available)
# plt.style.use('fivethirtyeight')
plt.style.use('seaborn-pastel')

def make_df_OHLC_symbol(symbolname,start_date,end_date):
    df = web.DataReader(symbolname,data_source = 'yahoo', start = start_date, end = end_date)
    df.rename(columns = {'Close': 'close',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Volume':'volume'}, inplace =True)
    return df

def data_selector(symbolname, years, column_name):
    today = datetime.today().strftime('%Y-%m-%d')
    start_date_calulated = datetime.today() - timedelta(days=years * 365)
    start_date = start_date_calulated.strftime('%Y-%m-%d')

    print('Modelling for {} index time series for simplicity for {} year(s)'.format(symbolname, years))
    new_or_old = input("Download new data (new) or use saved data (saved)? : ").lower()

    if new_or_old == 'new':
        print('Start date set to {} and end date to {}'.format(start_date, today))
        print('Getting OHLC for {} from {} to {}'.format(symbolname, start_date, today))
        df_symbol = make_df_OHLC_symbol(symbolname, start_date, today)
        file_name = 'All_Exports/01_Downloaded_Data/' + symbolname + '_input.csv'
        df_symbol.to_csv(file_name)
    elif new_or_old == 'saved':
        file_name = 'All_Exports/01_Downloaded_Data/' + symbolname + '_input.csv'
        df_symbol = pd.read_csv(file_name)
        print('Loading existing data: {}'.format(file_name))
    else:
        print("Error, please check filename or check for valid input")
    df_symbol.reset_index(inplace=True)
    df_symbol['Date'] = pd.to_datetime(df_symbol['Date'])
    df_symbol = df_symbol[['Date', column_name]].drop_duplicates()
    df_symbol.set_index('Date', inplace=True)

    return df_symbol

def make_time_x(data, use_of_price, lookbacklen):
    df = data.filter([use_of_price])
    dataset = df.values
    x = []

    for i in range(lookbacklen, len(dataset)):
        x.append(dataset[i-lookbacklen:i, 0])

    x = np.array(x)

    dfx = pd.DataFrame(x)
    dfx[use_of_price] = dfx[lookbacklen-1].shift(-1)
    minus_len = -1*dfx.shape[0]
    dfx.set_index(data.index[minus_len:], inplace=True, drop=True)
    dfx.dropna(inplace = True)
    dfx.reset_index(inplace=True, drop=True)
    return dfx

def train_test_split(data, split_ratio, column_name):

    x_variables = list(data.iloc[:, :-1].columns) # Choosing x variables

    data_validation = data[-20:].copy() # Last 20 rows for validation
    data_test_train = data[:-20].copy() # rows zero to end minus 20 choosen in validation

    # Splitting randomly
    selection = np.random.rand(len(data_test_train)) < split_ratio
    df_train, df_test = data_test_train[selection], data_test_train[~selection]

    # Creating all x array
    x_test = df_test[x_variables].values
    x_train = df_train[x_variables].values
    x_val = data_validation[x_variables].values

    # Creating all y array
    y_test = df_test[column_name].values
    y_train = df_train[column_name].values
    y_val = data_validation[column_name].values

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    scaler = [mean, std]

    x_test = (x_test - mean) / std
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    return x_test, y_test, x_train, y_train, x_val, y_val, scaler

##### CNN1d Functions #####
def reshape_for_CNN1D(x_test, x_train, x_val):

    input_dimension = 1

    sample_size = x_test.shape[0]
    time_steps = x_test.shape[1]
    x_test = x_test.reshape(sample_size, time_steps, input_dimension)

    sample_size = x_train.shape[0]
    time_steps = x_train.shape[1]
    x_train = x_train.reshape(sample_size, time_steps, input_dimension)

    sample_size = x_val.shape[0]
    time_steps = x_val.shape[1]
    x_val = x_val.reshape(sample_size, time_steps, input_dimension)

    return x_test, x_train, x_val
def build_conv1D_model(x_train, patience):
    n_timesteps = x_train.shape[1]
    n_features = x_train.shape[2]

    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))

    model.add(Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1",  input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', name="Conv1D_3"))
    model.add(MaxPooling1D(2, padding='same', name="MaxPooling1D"))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', name="Dense_1"))
    model.add(Dense(n_features, name="Dense_2"))

    optimizer = RMSprop(0.001)
    loss = 'mean_squared_error'
    metrics = ['accuracy', 'mae']

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    print(model.summary())

    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=patience)

    best_model_save_location = 'All_Exports/02_Exported_Models/Best_Models/best_model_cnn' + ".h5"

    mc = ModelCheckpoint(best_model_save_location,
                         monitor='val_accuracy',
                         mode='max',
                         verbose=0,
                         save_best_only=True)

    print(model.summary())

    return model, es, mc
###########################

##### LSTM Functions #####
def reshape_for_LSTM(x_test, x_train, x_val):

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    return x_test, x_train, x_val
def LSTM_MODEL(x_train, patience):
    model = Sequential()
    n_batchsize = x_train.shape[1]
    n_features = 1
    model.add(Input(shape=(n_batchsize, n_features)))

    model.add(LSTM(60, return_sequences=True))
    model.add(LSTM(60, return_sequences=False))
    model.add(Dense(30))
    model.add(Dense(1))
    optimizer = 'adam'
    loss = 'mean_squared_error'
    metrics = ['accuracy', 'mae']
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=patience)

    best_model_save_location = 'All_Exports/02_Exported_Models/Best_Models/best_model_lstm' + ".h5"

    mc = ModelCheckpoint(best_model_save_location,
                         monitor='val_accuracy',
                         mode='max',
                         verbose=0,
                         save_best_only=True)

    print(model.summary())

    return model, es, mc
##########################

##### Running Keras Model Function #####
def run_model(x_train, y_train, model, epochs, use_validation, validation_split, batchsize, es, mc):
    if use_validation == 'n':
        history = model.fit(x_train, y_train,
                            batch_size=batchsize, epochs=epochs, verbose =1,
                            shuffle=True,
                            callbacks = [es,mc])
    elif use_validation == 'y':
        history = model.fit(x_train, y_train,
                            batch_size=batchsize, epochs=epochs, verbose =1,
                            validation_split=validation_split, shuffle=True,
                            callbacks = [es,mc])
    else:
        print('Enter y or n for use_validation')
    return model, history
########################################

def make_and_export_plots(column_name, history, model, x_val, y_val, plot_location_TV, plot_location_P, model_name):
    # TRAIN VAL PLOT
    plt_train_vs_val = plt
    plt_train_vs_val.figure(figsize=(fig_breadth, fig_height))
    plt_train_vs_val.plot(history.history['loss'])
    plt_train_vs_val.plot(history.history['val_loss'])
    plt_train_vs_val.title('Model train vs Validation loss - '+ model_name)
    plt_train_vs_val.ylabel('Loss')
    plt_train_vs_val.xlabel('Epoch')
    plt_train_vs_val.legend(['Train', 'Validation'], loc='upper right')
    plt_train_vs_val.savefig(plot_location_TV)
    del plt_train_vs_val

    # PREDICTION AND PLOT
    predictions = model.predict(x_val)
    rmse = np.sqrt(np.mean(((predictions - y_val) ** 2)))
    rmse = int(round(rmse, 0))
    print('RMSE is {}'.format(rmse))

    plt_prediction = plt
    plt_prediction.figure(figsize=(fig_breadth, fig_height))
    plt_prediction.plot(predictions)
    plt_prediction.plot(y_val)
    plt_prediction.title('Predictions on validation set - '+ model_name + ' - RMSE - '+str(rmse))
    plt_prediction.ylabel(column_name.title())
    plt_prediction.xlabel('Time')
    plt_prediction.legend(['prediction', 'validation'], loc='upper right')
    plt_prediction.savefig(plot_location_P)
    # plt_prediction.show()
    del plt_prediction