"""Measuring probes (DR-002). Never tuned: only the data changes.

hybrid: exact replica of aithin/model/hybrid_model.py:61-89 (CNN -> BiLSTM -> Dense).
alt:    structurally different second probe (plain CNN + global average pooling), used at milestones.
"""
import keras
from keras import layers

INPUT_SHAPE = (128, 128, 1)


def build_hybrid(num_classes: int = 20) -> keras.Model:
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Reshape((16, 16 * 128))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="hybrid")


def build_alt(num_classes: int = 20) -> keras.Model:
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = inputs
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="alt")


PROBES = {"hybrid": build_hybrid, "alt": build_alt}


def build(probe: str, num_classes: int = 20, lr: float = 1e-3) -> keras.Model:
    model = PROBES[probe](num_classes)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
