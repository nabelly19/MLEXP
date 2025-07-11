from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.Input(9),
        layers.Dense(36, activation='relu'),
        layers.Dense(36, activation='relu'),
        layers.Dense(9, activation='linear'),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
