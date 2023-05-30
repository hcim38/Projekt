import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.utils import to_categorical
import os


#global vars
image_size = (64, 64)
batch_size = 32
number_of_categories = 43
training_epochs = 3


def createModel():
    model = keras.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=image_size + (3,)))
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
    model.add(Dense(number_of_categories, activation='softmax'))
    # Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model
def trainAndSaveModel(model, train_ds, val_ds):
    anc = model.fit(train_ds, batch_size=batch_size, epochs=training_epochs, validation_data=val_ds)
    model.save("my_model.h5")
    return model

def testTrainedModel():

    return

def loadDataForTrainig():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset",
        label_mode='categorical',
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds, val_ds

if __name__ == '__main__':

    while True:
        print("What would you like to do?")
        print("1. Load dataset and train new model")
        print("2. Load trained model")
        print("3. Recognize signs")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            train_ds, val_ds = loadDataForTrainig()
            Model = createModel()
            trainAndSaveModel(Model, train_ds, val_ds)
        elif choice == "2":
            Model = keras.models.load_model("my_model.h5")
        elif choice == "3":
            #TODO recognize_signs()
            print("WIP")
        elif choice == "4":
            print("Exiting the program...")
            break
        else:
            print("Invalid choice. Please enter a valid option (1-4).")