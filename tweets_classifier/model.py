import tensorflow as tf
from tensorflow import keras
from .layers import *

input = keras.layers.Input(shape = (None,), name = "Input", dtype = "int64")
positional = PositionalEmbedding(165, 20_000, 256)(input)
encoder = TransformerEncoder(256, 32, 8)(positional)
pooling = keras.layers.GlobalMaxPooling1D()(encoder)
dropout = keras.layers.Dropout(0.5)(pooling)
output = keras.layers.Dense(1, activation = "sigmoid")(dropout)

model = keras.Model(inputs = input, outputs = output)

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 1e-5, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9),
    loss = keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.Precision(), keras.metrics.AUC()]
)