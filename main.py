import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.utils import to_categorical
import os


def trainFromData():

    return

def testTrainedModel():

    return

def loadDataForTrainig():

    return

if __name__ == '__main__':
    #testing
    image_size = (64, 64)
    batch_size = 32

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset",
        label_mode='categorical',
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)

    #
    # Building the model
    # TODO ogarnac jak to dziala i podzielic na mniejsze funkcje xD
    model = keras.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    # Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    print("test xd")
    anc = model.fit(train_ds, batch_size=32, epochs=3, validation_data=val_ds)
    model.save("my_model.h5")