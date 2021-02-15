'''
-----------------------------------------------------------------------
Created by Manasvi Mohan Sharma on 15/02/21 (dd/mm/yy)
Project Name: ML_Price_Prediction | File Name: main.py
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
This is the main file.
-----------------------------------------------------------------------
'''

import custom_functions
import sys

symbolname = "^NSEI"
# symbolname = "^NSEBANK"
years = 5
timeframe = '1d'
column_name = 'close'
lookbacklen = 21 # How many x variables would be required
period = 5 # How much in future is to be predicted

# Download or Load data
df_symbol = custom_functions.data_selector(symbolname, years, column_name)
data = custom_functions.make_time_x(df_symbol, column_name, lookbacklen)

data[column_name] = data[column_name].shift(-(period - 1))  # Here -1 is done because y_values are already 1 day ahead
data.dropna(inplace=True)

split_ratio = 0.8
x_test, y_test, x_train, y_train, x_val, y_val, scaler = custom_functions.train_test_split(data, split_ratio, column_name)

del data, df_symbol

epochs = 5000
batchsize = int(lookbacklen*2)
use_validation = 'y'
validation_split = 0.2
patience = 50

# Choose and make model
which_model_to_make = input('Choose between LSTM (1) and CNN 1d (2). Type 1 or 2: ')

if which_model_to_make == '1':
    x_test, x_train, x_val = custom_functions.reshape_for_LSTM(x_test, x_train, x_val)
    model, es, mc = custom_functions.LSTM_MODEL(x_train, patience)
    plot_location_TV = 'All_Exports/03_Model_Logs/Plots/01_LSTM_Train_vs_Validation.png'
    plot_location_P = 'All_Exports/03_Model_Logs/Plots/02_LSTM_Prediction.png'
    model_name = 'LSTM'
elif which_model_to_make == '2':
    x_test, x_train, x_val = custom_functions.reshape_for_CNN1D(x_test, x_train, x_val)
    model, es, mc = custom_functions.build_conv1D_model(x_train, patience)
    plot_location_TV = 'All_Exports/03_Model_Logs/Plots/01_CNN1D_Train_vs_Validation.png'
    plot_location_P = 'All_Exports/03_Model_Logs/Plots/02_CNN1D_Prediction.png'
    model_name = 'CNN1D'
else:
    print('Invalid Input')
    sys.exit()

# Train Model
model, history = custom_functions.run_model(x_train, y_train,
                                            model, epochs,
                                            use_validation, validation_split,
                                            batchsize, es, mc)

# Export train vs val plot and predictions
custom_functions.make_and_export_plots(column_name, history, model, x_val, y_val, plot_location_TV, plot_location_P, model_name)