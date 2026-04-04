"""Keras model definitions for perception.

Migrated incrementally from the legacy training models.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Convolution2D, Dense, Dropout,
                                     Flatten, Input, MaxPooling2D)
from tensorflow.keras.optimizers import Adam


def optimal_categorical(resolution: Tuple[int, int], use_smooth: bool) -> Model:
    """Optimized categorical model (steering classification + throttle regression)."""
    relu = tf.keras.layers.ReLU()
    img_in = Input(shape=(resolution[0], resolution[1], 3), name="img_in")
    x = Convolution2D(24, (3, 3), strides=(2, 2))(img_in)
    x = BatchNormalization(axis=1)(x)
    x = relu(x)
    x = Dropout(.2)(x)

    x = Convolution2D(32, (3, 3), strides=(2, 2))(x)
    x = BatchNormalization(axis=1)(x)
    x = relu(x)
    x = Dropout(.2)(x)

    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = relu(x)
    x = Dropout(.2)(x)

    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = relu(x)
    x = Dropout(.2)(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(.2)(x)
    x = Dense(50, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(.2)(x)

    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
    outputs = [angle_out, throttle_out]

    if use_smooth:
        sth_out = Dense(15, activation='softmax', name='sth_out')(x)
        outputs.append(sth_out)

    model = Model(inputs=[img_in], outputs=outputs)
    optimizer = Adam(learning_rate=0.0005)
    loss = {'angle_out': 'categorical_crossentropy', 'throttle_out': 'mean_absolute_error'}
    loss_weights = {'angle_out': 0.9, 'throttle_out': .01}

    if use_smooth:
        loss['sth_out'] = 'categorical_crossentropy'
        loss_weights['sth_out'] = 0.9

    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
    return model


def default_categorical(resolution: Tuple[int, int], use_smooth: bool) -> Model:
    """Default categorical model (steering classification + throttle regression)."""
    img_in = Input(shape=(resolution[0], resolution[1], 3), name='img_in')
    x = Convolution2D(48, (3, 3), strides=(2, 2), activation='relu')(img_in)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.2)(x)

    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    throttle_out = Dense(15, activation='relu', name='throttle_out')(x)
    outputs = [angle_out, throttle_out]

    if use_smooth:
        sth_out = Dense(15, activation='softmax', name='sth_out')(x)
        outputs.append(sth_out)

    model = Model(inputs=[img_in], outputs=outputs)
    loss = {'angle_out': 'categorical_crossentropy', 'throttle_out': 'mean_absolute_error'}
    loss_weights = {'angle_out': 0.9, 'throttle_out': .01}

    if use_smooth:
        loss['sth_out'] = 'categorical_crossentropy'
        loss_weights['sth_out'] = 0.9

    model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights)
    return model
