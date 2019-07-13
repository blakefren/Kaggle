"""
This is for Kaggle's Digit Recognizer training competition:
https://www.kaggle.com/c/digit-recognizer/data

Written by Blake French
blakefren.ch
"""


import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
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
    data = data / 255.0
    
    # Reshape images.
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)
    
    return data


def prep_model():
    
    print('Prepping model...\n')

    model = Sequential()

    # Add layers : 3 Conv2D, 1 Flat, 2 Dense.
    # TODO: find the correct layers/parameters for what I want
    model.add(Conv2D(
        24,
        kernel_size=3,
        activation='relu',
        input_shape=(img_rows, img_cols, 1)
    ))
    model.add(Conv2D(
        36,
        kernel_size=3,
        activation='relu'
    ))
    model.add(Conv2D(
        48,
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
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, train_data, categories):
    
    print('Training model...\n')

    batch_size = 100
    num_examples = len(train_data)
    validation_split = 0.2  # 20% validation split.

    # Randomly split the train/validation data sets.
    t_data, v_data, t_cats, v_cats = train_test_split(
        train_data,
        categories,
        train_size=(1-validation_split),
        random_state=1138)

    # Create generator for training data.
    gen_a = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range=0.1,  # Adding zoom out causes vey low accuracy.
        rotation_range=10)
    train_gen = gen_a.flow(
        t_data,
        t_cats,
        batch_size=batch_size)

    # Fit the model.
    # TODO : add a learning rate reducer (what is this called?)
    model.fit_generator(
        train_gen,
        steps_per_epoch=(num_examples // batch_size),
        epochs=50,
        validation_data=(v_data, v_cats))

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
        del train_data
        del train_categories

    # Make our predictions.
    test_data = read_data(test_file)
    test_data = prep_data(test_data)
    preds = make_predictions(digit_model, test_data)
    save_predictions(preds)
