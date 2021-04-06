import numpy as np
import datetime
from typing import List, Tuple
from math import ceil, sqrt
from tensorflow import keras
from tensorflow.keras import layers
from parser import InputParser
from model import model_A
from preprocessing import DataNormalizer
from lm import levenberg_marquardt as lm

def split_train_test(X: List, Y: List, ratio: float = 0.8) -> Tuple:
    train_count = ceil(len(X) * ratio)
    return (np.array(X[:train_count], dtype=np.float32),
            np.array(Y[:train_count], dtype=np.float32),
            np.array(X[train_count:], dtype=np.float32),
            np.array(Y[train_count:], dtype=np.float32))

def get_test_set_predictions(X: List, model: keras.Model) -> List:
    Y = []
    for el in X:
        prediction = model.predict(np.reshape(el,(1,-1))) 
        Y.extend(prediction)
    return Y

def get_mean_neighbour_distances(neighbours: List, Y: List, offset: int = 0) -> List:
    distances = []
    for idx, y in enumerate(Y):
        for n in neighbours[idx + offset]:
            if n == idx:
                continue
            distances.append(sqrt((Y[n - offset][0] - y[0]) ** 2 +
                                  (Y[n - offset][1] - y[1]) ** 2 +
                                  (Y[n - offset][2] - y[2]) ** 2))
    return distances

def get_test_distances_mse(neighbours: List, Y_pred: List, Y_test: List, offset: int = 0) -> float:
    test_distances = get_mean_neighbour_distances(neighbours, Y_test, len(Y_train))
    pred_distances = get_mean_neighbour_distances(neighbours, Y_pred, len(Y_train))
    dist_sum = 0
    for i in range (0, len(test_distances)):
        dist_sum += (test_distances[i] - pred_distances[i])**2
    return dist_sum/len(test_distances)


if __name__ == "__main__":
    input_type = 'A'
    activation_function = 'linear'
    token_increment = 1.0
    batch_size = 128
    epochs = 250
    train_test_ratio = 0.9
    use_lm = False

    parser = InputParser(f'data_nmr/H{input_type}_nmr.inp',
                         input_type='B', token_increment=token_increment)
    # parser = InputParser(f'data_extended/H{input_type}_All.inp',
    #                      input_type=input_type, token_increment=token_increment)
    data = parser.parse_input()
    norm = DataNormalizer(data)
    norm_data = norm.get_normalized_dateset()
    X, Y = parser.split_input_and_output(data)
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y, ratio=train_test_ratio)

    print('Data sample')
    for i in range (0, 20):
        print(f'Input: {X[i]} ### Output {Y[i]}')
    print(f'Train data count: {len(X_train)}\n Test data count: {len(X_test)}')

    if use_lm: # TODO Document requirements for Levenberg Marquardt
        model = lm.ModelWrapper(model_A(activation_function=activation_function))
        model.compile(loss=lm.MeanSquaredError(), optimizer=keras.optimizers.SGD(learning_rate=0.1),
                    metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.RootMeanSquaredError()])
    else:
        model = model_A(activation_function=activation_function)
        model.summary()
        model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.MeanAbsoluteError(), 
                                                        keras.metrics.RootMeanSquaredError()])
    log_dir = 'tb_logs/' + f'{input_type}_{batch_size}_' + \
              f'{epochs}_{activation_function}_token-inc{token_increment}' 
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
              callbacks=[tensorboard_callback])

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(f'###########\nTest set: \nMSE: {score[0]}\nMAE: {score[1]}\nRMSE: {score[2]}\n') 
    print('###########\nSample predictions (first 10 elements of test set):')

    # with open(f'results/{input_type}_LM_{activation_function}_extended_data_no_norm.txt', 'w+') as f:
    #     f.write(f'MSE: {score[0]} MAE: {score[1]} RMSE: {score[2]}\n')
    #     for i in range(0, len(X_test)):
    #         predicted_y = model.predict(np.reshape(X_test[i],(1,-1)))
    #         _, out_test_y = norm.inverse_scale_row(list(X_test[i]), list(Y_test[i]))
    #         _, out_pred_y = norm.inverse_scale_row(list(X_test[i]), predicted_y.tolist()[0])
    #         #f.write(f'{Y_test[i]}{predicted_y[0]}\n')
    #         f.write(f'{out_test_y}{out_pred_y}\n')

    for i in range (0, 10):
        predicted_y = model.predict(np.reshape(X_test[i],(1,-1)))
        print(f'True value: {Y_test[i]}, predicted value {predicted_y}') 

        #_, out_test_y = norm.inverse_scale_row(list(X_test[i]), list(Y_test[i]))
        #_, out_pred_y = norm.inverse_scale_row(list(X_test[i]), predicted_y.tolist()[0])
        #print(f'True value: {out_test_y}, predicted value {out_pred_y}')

    