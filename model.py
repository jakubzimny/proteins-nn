from tensorflow import keras
from tensorflow.keras import layers

def model_AB(in_shape: int = 5, out_shape: int = 3) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(5,)),
            layers.Dense(5, activation="linear"),
            layers.Dense(6, activation="linear"),
            layers.Dense(7, activation="linear",),
            layers.Dense(7, activation="linear"),
            layers.Dense(8, activation="linear"),
            layers.Dense(8, activation="linear"),
            layers.Dense(9, activation="linear"),
            layers.Dense(8, activation="linear"),
            layers.Dense(8, activation="linear"),
            layers.Dense(7, activation="linear"),
            layers.Dense(7, activation="linear"),
            layers.Dense(6, activation="linear"),
            layers.Dense(5, activation="linear"),
            layers.Dense(3, name="out"),
        ]
    )

    return model

    