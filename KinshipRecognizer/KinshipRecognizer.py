"""
This is for Kaggle's Northeastern SMILE Lab - Recognizing Faces in the Wild playground competition:
https://www.kaggle.com/c/recognizing-faces-in-the-wild


The general model will be to create feature vectors of each face, then
compare their Euclidean distance to get a value.
I think I might use a logistic regression model to make the final prediction.
It will determine the "distance" that two faces must be to be kin.


Written by Blake French
blakefren.ch
"""


import os
import sys
import csv
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras_vggface.vggface import VGGFace  # Installed from pip via git, not direct from pip.


# Define a bunch of stuff. -------------------------------------
dataset_dir = os.path.join('E:\\', 'datasets', 'kaggle-recognizing-faces-in-the-wild')
test_files = os.path.join(dataset_dir, 'test')
train_files = os.path.join(dataset_dir, 'train')
submission_pairs_file = os.path.join(dataset_dir, 'sample_submission.csv')
train_relationships_file = os.path.join(dataset_dir, 'train_relationships.csv')
feature_vector_file = os.path.join(dataset_dir, 'VGGFace_feature_vectors.csv')
output_file = '.\\predictions.csv'
model_file = 'comparison_model_weights.h5'

img_rows = 224
img_cols = 224
img_channels = 3
num_categories = 2  # Kin, or not-kin
# --------------------------------------------------------------


def read_csv_file(filename):
    
    print('Reading data from ' + filename + '\n')
    
    if not os.path.exists(filename):
        return None

    return pd.read_csv(filename)


def picture_to_tensor(filename):
    
    img = img_to_array(load_img(filename))  # Returns 3D np array.
    img = img / 255.0  # Feature scaling.

    return img.reshape([1, img_rows, img_cols, img_channels])


def get_feature_vectors(model, file_paths):

    print('Creating feature vectors...')
    
    start = time.time()
    features = {}
    for file in file_paths:
        feat_vec = model.predict(picture_to_tensor(file))[0]  # Returns array of arrays with one element
        num_features = len(feat_vec)
        split_path = file.split(os.sep)
        family, person, pic = split_path[len(split_path) - 3:]
        key = os.path.join(family, person, pic)
        features[key] = feat_vec
        # features[os.path.basename(file)] = feat_vec
        
    print('Calculation time elapsed: ' + str(int(time.time() - start)) + ' seconds.\n')
    
    return features, num_features


def save_feature_vectors(feature_vectors, num_features):

    print('Saving feature vectors to file...\n')

    with open(feature_vector_file, 'w', newline='') as csvfile:
        
        w = csv.writer(csvfile)
        w.writerow(['ImageFileName'] + ['Feature'+str(i) for i in range(num_features)])
        
        for key in feature_vectors.keys():
            w.writerow([key] + feature_vectors[key].tolist())


def prep_model(num_features):
    
    print('Prepping model...\n')

    model = Sequential()

    model.add(Dense(
        256,
        activation='relu',
        input_shape=(num_features,)
    ))
    model.add(Dense(
        32,
        activation='relu'
    ))
    model.add(Dropout(0.2))
    model.add(Dense(
        32,
        activation='relu'
    ))
    model.add(Dropout(0.2))
    model.add(Dense(  # Output layer.
        num_categories,  # kin or not kin.
        activation='softmax'
    ))
    
    # Compile the model.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, feature_dict, relation_dict):

    '''
    Training plan (76613631 pairs total):
    1. Select 100000 random pairs of images
        - pairs cannot use two images from the same individual
        - remove images from list after selection
        - how do I ensure that each batch has the same ratio of pos/neg examples?
    2. Create data and labels for these
    3. Call model.fit to train on these 100000 pairs
        - use 80/20 validation split
    4. Repeat steps 1-3 ~760 times to train incrementally for all 76 million image pairs
        - estimate this will take 16GB of ram per batch to hold training set
    '''
    
    print('Training model...\n')

    return model


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
    
    '''
    Initial plan:
    1. Use pre-trained face model - keras_vggface
    2. Iterate through all training faces to create their feature vectors
    3. Create smaller NN where the inputs are the raw differences of the feature vectors
    4. Train NN with ALL facial combinations, noting which are actually related
        -there are ~76 million combinations...do I need all? (Shia LaBeouf it)
        -what if I just use one from each person instead of all photos? (~3.8 million combinations)
    5. Run all test images through CNN to generate their feature vectors
    6. Run all test feature vectors/distances through NN model to predict kinship
    '''

    # TODO : add object cleanup as we go
    # Suppress tensorflow INFO, WARNING, and ERROR messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load pre-trained facial model. Step 1.
    vggface = VGGFace()  # Uses default vgg16 model.
    print()

    # Get and prep training images and create feature vectors. Step 2.
    train_relationships = read_csv_file(train_relationships_file)
    train_rels_dict = defaultdict(list)
    print('Gathering kin relationships...\n')
    for i, row in train_relationships.iterrows():  # Store kin relationships for quick reference.
        key = os.path.join(train_files, row['p1'].replace('/', os.sep))
        value = os.path.join(train_files, row['p2'].replace('/', os.sep))
        train_rels_dict[key].append(value)
        train_rels_dict[value].append(key)
        # We can now get kin relations for all training images by:
        # train_rels_dict.get(image_path, [])

    # Check to see if the feature vectors have already been calculated.
    feature_vectors = read_csv_file(feature_vector_file)
    num_features = -1
    if feature_vectors is None:
        print('Feature vector file not found.\n')
        training_image_paths = []
        for folder in os.walk(train_files):  # Get all training image paths.
            training_image_paths.extend([os.path.join(folder[0], file) for file in folder[2]])
        feature_vectors, num_features = get_feature_vectors(vggface, training_image_paths)
        save_feature_vectors(feature_vectors, num_features)
        del training_image_paths
    else:  # Convert to dict.
        print('Converting stored feature vector...\n')
        labels = feature_vectors['ImageFileName']
        num_features = len(feature_vectors.columns) - 1
        feature_vectors = feature_vectors.drop('ImageFileName', axis=1).to_numpy()
        fv_dict = {labels[i]: feature_vectors[i] for i in range(len(feature_vectors))}
        feature_vectors = fv_dict
        del fv_dict, labels

    # Create comparison model. Step 3.
    comp_model = prep_model(num_features)

    # Train comparison model. Step 4.
    comp_model = train_model(comp_model, feature_vectors, train_rels_dict)

    sys.exit()  # TEMP

    # Get and prep test images and create feature vectors. Step 5.
    test_image_paths = []
    for folder in os.walk(test_files):  # Get all training image paths.
        test_image_paths.extend([os.path.join(folder[0], file) for file in folder[2]])
    feature_vectors = get_feature_vectors(vggface, test_image_paths)

    # Make predictions. Step 6.
    pairs = read_csv_file(submission_pairs_file)['img_pair']
    preds = make_predictions(comp_model)
    save_predictions(preds)
