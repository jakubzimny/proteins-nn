import numpy as np
import datetime
import time
import tensorflow as tf
from typing import List, Tuple
from math import ceil, sqrt
from tensorflow import keras
from tensorflow.keras import layers
from parser import InputParser
from model import model_ff, model_lstm
from preprocessing import DataNormalizer
from lm import levenberg_marquardt as lm
from utils import split_train_test


if __name__ == "__main__":
    input_type = 'A'
    activation_function = 'selu'
    token_increment = 10
    token_init = 10
    batch_size = 1024
    epochs = 100
    train_test_ratio = 0.9
    use_lm = False
    lstm = True

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print('No GPU found by TensorFlow')
        exit(0)

    parser = InputParser(f'data_nmr/HA_nmr.inp',
                        input_type=input_type, token_increment=token_increment, token_init=token_init)
    data = parser.parse_input()
    norm = DataNormalizer(data)
    print(data[:10])
    norm_data = norm.get_normalized_dateset()
    X, Y = parser.split_input_and_output(data)
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y, ratio=train_test_ratio)

    print('Data sample')
    for i in range (0, 20):
        print(f'Input: {X[i]} ### Output {Y[i]}')
    print(f'Train data count: {len(X_train)}\n Test data count: {len(X_test)}')
    
    start = time.time()
    if lstm:
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        model = model_lstm(shape=(X_train.shape[1], X_train.shape[2]), activation_function=activation_function)
        model.summary()
        model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.MeanAbsoluteError(), 
                                                            keras.metrics.RootMeanSquaredError()])
    else:
        if use_lm: 
            model = lm.ModelWrapper(model_ff(activation_function=activation_function))
            model.compile(loss=lm.MeanSquaredError(), optimizer=keras.optimizers.SGD(learning_rate=0.1),
                        metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.RootMeanSquaredError()])
        else:
            model = model_ff(activation_function=activation_function)
            model.summary()
            model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.MeanAbsoluteError(), 
                                                                 keras.metrics.RootMeanSquaredError()])
    log_dir = 'tb_logs/LSTM_A_nmr' + f'_bs_{batch_size}_e' + \
              f'{epochs}_token-inc{token_increment}_init{token_init}_{activation_function}_split{train_test_ratio}' 
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
              callbacks=[tensorboard_callback])

    end = time.time()
    print(f'###########\nTraining time: {end - start}s')

    score = model.evaluate(X_test, Y_test, verbose=2)
    print(f'###########\nTest set: \nMSE: {score[0]}\nMAE: {score[1]}\nRMSE: {score[2]}') 
    print('###########\nSample predictions (first 10 elements of test set):')
   

    if lstm:
        predicted = model.predict(X_test)
        for i in range(0,50):
            print(f'True value: {Y_test[i]}, predicted value {predicted[i]}')


    model.save(f'models/model.h5')