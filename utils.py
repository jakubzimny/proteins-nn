from typing import List, Tuple
from tensorflow import keras
import numpy as np
from math import ceil, sqrt

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

