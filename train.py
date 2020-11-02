import numpy as np
import datetime
from typing import List, Tuple
from math import ceil
from tensorflow import keras
from tensorflow.keras import layers
from parser import InputParser
from model import model_A

def split_train_test(X: List, Y: List, ratio: float = 0.8) -> Tuple:
    train_count = ceil(len(X) * ratio)
    return (np.array(X[:train_count], dtype=np.float32),
            np.array(Y[:train_count], dtype=np.float32),
            np.array(X[train_count:], dtype=np.float32),
            np.array(Y[train_count:], dtype=np.float32))

if __name__ == "__main__":
    input_type = 'A'
    activation_function = 'selu'
    token_increment = 1.0
    batch_size = 128
    epochs = 100

    parser = InputParser(f'proteinParams/data/H{input_type}_All.inp',
                         input_type=input_type, token_increment=token_increment)
    whole_dataset = parser.parse_input()
    X, Y = parser.split_input_and_output(whole_dataset)
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y, ratio=0.85)
    print('Data sample')
    for i in range (0, 20):
        print(f'Input: {X[i]} ### Output {Y[i]}')
    print(f'Train data count: {len(X_train)}\n Test data count: {len(X_test)}')

    model = model_A(activation_function=activation_function)
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.MeanAbsoluteError()])

    log_dir = 'tb_logs/' + f'{input_type}_{batch_size}_' + \
              f'{epochs}_{activation_function}_token-inc{token_increment}' 
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
              callbacks=[tensorboard_callback])

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(f'###########\nTest set: \nMSE: {score[0]}\nMAE: {score[1]}')
    print('###########\nSample predictions (first 10 elements of test set):')
    for i in range (0, 10):
        predicted_y = model.predict(np.reshape(X_test[i],(1,-1)))
        print(f'True value: {Y_test[i]}, predicted value {predicted_y}')