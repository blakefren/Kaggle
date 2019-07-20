"""
This is for Kaggle's Northeastern SMILE Lab - Recognizing Faces in the Wild playground competition:
https://www.kaggle.com/c/recognizing-faces-in-the-wild

Written by Blake French
blakefren.ch
"""


import os
import csv
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout


# Define a bunch of stuff. -------------------------------------
dataset_dir = os.path.join('C:', 'datasets', 'kaggle-recognizing-faces-in-the-wild')
test_files = os.path.join(dataset_dir, 'test')
train_files = os.path.join(dataset_dir, 'train')
submission_pairs_file = os.path.join(dataset_dir, 'sample_submission.csv')
train_relationships_file = os.path.join(dataset_dir, 'train_relationships.csv')
output_file = 'predictions.csv'
model_file = 'model_weights.h5'

img_rows = 224  # TODO : should I reduce image size? (probably)
img_cols = 224
img_channels = 1  # TODO : should I reduce to black and white? (prabably not)
num_categories = 2  # Kin, or not-kin
# --------------------------------------------------------------


def read_model():

    print('Reading previous model...')
    if os.path.exists(model_file):
        print()
        return load_model(model_file)
    else:
        print('Previous model not found.\n')
        return None


def read_csv_file(filename):
    
    print('Reading data...\n')

    # Read data from file
    return pd.read_csv(filename)
    # return np.loadtxt(filename, skiprows=1, delimiter=',')


def read_picture(filename):
    # TODO : do I need this, or is there a way to do them all at once?
    pass


def prep_model():
    
    print('Prepping model...\n')

    model = Sequential()

    # TODO : create the model

    return model


def train_model():

    # TODO
    pass


def save_model(model):

    print('Saving model to file...\n')

    if not os.path.exists(model_file):
        model.save(model_file)


def make_predictions(model, images, image_pairs):
    
    print('Making predictions...\n')
    preds = []
    
    for pair in image_pairs:

        image_1, image_2 = pair.split('-')
        image_1 = os.path.join(test_files, image_1)
        image_2 = os.path.join(test_files, image_2)

        # TODO : make prediction for each image pair

        preds.append(pair, prediction)
    
    return preds


def save_predictions(preds):
    
    # preds is a list of 2-tuples where the first
    # item is the image pair and the second
    # is the 0/1 prediction of kinship.

    print('Saving predictions...\n')

    with open(output_file, 'w', newline='') as csvfile:
        
        w = csv.writer(csvfile)
        w.writerow(['img_pair', 'is_related'])
        
        # TODO : figure out my labels
        # TODO : write to the file


if __name__ == '__main__':
    
    print()
    # Check for existing model.
    kin_model = read_model()

    if kin_model is None:
        
        # Prep training images.
        # TODO : get the data ready

        # Train the model.
        kin_model = prep_model()
        kin_model = train_model()  # TODO : inputs
        save_model(kin_model)

        # TODO : del unneeded objects
    
    # Make predictions.
    # TODO : prep/gather images
    pairs = read_csv_file(submission_pairs_file)['img_pair']
    preds = make_predictions(kin_model, images, pairs)
    save_predictions(preds)
