"""
This is for Kaggle's Digit Recognizer training competition:
https://www.kaggle.com/c/digit-recognizer/data

Written by Blake French
blakefren.ch
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D


# Define a bunch of stuff. -------------------------------------
training_file = '.\\digit-recognizer\\train.csv'
test_file = '.\\digit-recognizer\\test.csv'
output_file = '.\\digit-recognizer\\predictions.csv'
img_rows = 28
img_cols = 28
num_categories = 10  # Nums 0-9
# --------------------------------------------------------------


def read_data(filename):
    
    print('\nReading data...\n')

    # Read data from file
    return np.loadtxt(filename, skiprows=1, delimiter=',')


def prep_data(data):

    print('Prepping data...\n')

    # Perform feature scaling.
    data = data / 255
    
    # Reshape images.
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)
    
    return data


def prep_model():
    
    print('Prepping model...\n')

    model = Sequential()

    # Add layers : 3 Conv2D, 1 Flat, 2 Dense.
    # TODO: find the correct parameters for what I want
    model.add(Conv2D(
        12,
        kernel_size=3,
        activation='relu',
        input_shape=(img_rows, img_cols, 1)
    ))
    model.add(Conv2D(
        12,
        kernel_size=3,
        activation='relu'
    ))
    model.add(Conv2D(
        12,
        kernel_size=3,
        activation='relu'
    ))
    model.add(Flatten())
    model.add(Dense(
        100,
        activation='relu'
    ))
    model.add(Dense(
        num_categories,
        activation='softmax'
    ))
    
    # Compile the model.
    # TODO: find the correct parameters for what I want
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def train_model(model, train_data, categories):
    
    print('Training model...\n')

    # TODO : figure out if these are the args I want.
    model.fit(
        train_data,
        categories,
        batch_size=100,  # Mini-batch gradient descent
        epochs=4,
        validation_split=0.2  # 20% for validation
    )

    return model


def make_predictions(model, test_data):

    print('Making predictions...\n')

    return model.predict(test_data)


def save_predictions(preds):
    
    print('Saving predictions...\n')

    # TODO : all of it
    print(preds)  # TEMP


if __name__ == '__main__':
    
    # Train the model.
    train_data = read_data(training_file)
    train_categories = keras.utils.to_categorical(train_data[:, 0], num_categories)
    train_data = train_data[:, 1:]
    train_data = prep_data(train_data)
    digit_model = prep_model()
    digit_model = train_model(digit_model, train_data, train_categories)

    # Make our predictions.
    test_data = read_data(test_file)
    test_data = prep_data(test_data)
    preds = make_predictions(digit_model, test_data)
    save_predictions(preds)
