from tensorflow import keras
from tensorflow.keras import layers

def model_A(in_shape: int = 5, out_shape: int = 3, activation_function: str = 'linear') -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(5,)),
            #layers.BatchNormalization(),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(10, activation=activation_function),
            layers.Dense(3, name="out"),
        ]
    )

    return model