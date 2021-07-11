from tensorflow import keras
from tensorflow.keras import layers

def model_ff(in_shape: int = 5, out_shape: int = 3, activation_function: str = 'linear') -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(14,)),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(30, activation=activation_function),
            layers.Dense(3, name="out"),
        ]
    )
    return model


def model_lstm(shape, activation_function: str = 'relu') -> keras.Model:
    model = keras.Sequential([
        layers.LSTM(32, input_shape=shape, activation=activation_function, return_sequences=True),
        layers.LSTM(32, input_shape=shape, activation=activation_function),
        layers.Dense(3, name="out"),
    ])
    return model