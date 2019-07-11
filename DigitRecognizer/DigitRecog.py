"""
This is for Kaggle's Digit Recognizer training competition:
https://www.kaggle.com/c/digit-recognizer/data

Written by Blake French
blakefren.ch
"""


import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D


# Define a bunch of stuff. -------------------------------------
training_file = '.\\digit-recognizer\\train.csv'
test_file = '.\\digit-recognizer\\test.csv'
output_file = '.\\digit-recognizer\\predictions.csv'
model_file = '.\\digit-recognizer\\model_weights.h5'
img_rows = 28
img_cols = 28
num_categories = 10  # Nums 0-9
# --------------------------------------------------------------


def read_model():

    print('Reading previous model...\n')
    if os.path.exists(model_file):
        return load_model(model_file)
    else:
        return None


def read_data(filename):
    
    print('Reading data...\n')

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


def save_model(model):

    print('Saving model to file...\n')

    if not os.path.exists(model_file):
        model.save(model_file)


def make_predictions(model, test_data):

    print('Making predictions...\n')

    return model.predict(test_data)


def save_predictions(preds):
    
    print('Saving predictions...\n')

    with open(output_file, 'w', newline='') as csvfile:
        
        w = csv.writer(csvfile)
        w.writerow(['ImageId', 'Label'])
        # The index is conveniently also the label.
        # argmax gets the index of the max value for each item.
        decoded = np.argmax(preds, axis=1)

        for i in range(len(decoded)):
            w.writerow([i+1, decoded[i]])  # Just the ImageID and label.


if __name__ == '__main__':
    
    print()
    # Check for previous model.
    digit_model = read_model()

    if digit_model is None:
        print('Previous model not found.\n')
        # Train the model.
        train_data = read_data(training_file)
        train_categories = keras.utils.to_categorical(train_data[:, 0], num_categories)
        train_data = train_data[:, 1:]
        train_data = prep_data(train_data)
        digit_model = prep_model()
        digit_model = train_model(digit_model, train_data, train_categories)
        save_model(digit_model)

    # Make our predictions.
    test_data = read_data(test_file)
    test_data = prep_data(test_data)
    preds = make_predictions(digit_model, test_data)
    save_predictions(preds)
