"""
This is for Kaggle's Northeastern SMILE Lab - Recognizing Faces in the Wild playground competition:
https://www.kaggle.com/c/recognizing-faces-in-the-wild


The general model will be to create feature vectors of each face, then
compare their Euclidean distance to get a value.
I will use a second NN to make the final prediction of the feature vector differences.
It will determine the "distance" that two faces must be to be kin.


Written by Blake French
blakefren.ch
"""


import os
import sys
import csv
import time
from itertools import combinations
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from keras_vggface.vggface import VGGFace  # Installed from pip via git, not direct from pip.


# Define a bunch of stuff. -------------------------------------
dataset_dir = os.path.join('E:\\', 'datasets', 'kaggle-recognizing-faces-in-the-wild')
test_files = os.path.join(dataset_dir, 'test')
train_files = os.path.join(dataset_dir, 'train')
submission_pairs_file = os.path.join(dataset_dir, 'sample_submission.csv')
train_relationships_file = os.path.join(dataset_dir, 'train_relationships.csv')
output_file = '.\\predictions.csv'
model_file = '.\\comparison_model_weights.h5'

img_rows = 224
img_cols = 224
img_channels = 3
num_categories = 2  # Kin, or not-kin
# --------------------------------------------------------------


def read_model(filename):

    print('Reading previous model...')
    if os.path.exists(filename):
        print()
        return load_model(filename)
    else:
        print('Previous model not found.\n')
        return None


def read_csv_file(filename):
    
    print('Reading data from ' + filename + '\n')
    
    if not os.path.exists(filename):
        return None

    return pd.read_csv(filename)


def picture_to_tensor(filename):
    
    img = img_to_array(load_img(filename))  # Returns 3D np array.
    img = img / 255.0  # Feature scaling.

    return img.reshape([1, img_rows, img_cols, img_channels])


def get_feature_vectors(model, file_paths, full_path_as_key):

    print('Creating feature vectors...')
    
    start = time.time()
    features = {}
    for file in file_paths:
        feat_vec = model.predict(picture_to_tensor(file))[0]  # Returns array of arrays with one element
        num_features = len(feat_vec)
        key = None
        split_path = file.split(os.sep)
        family, person, pic = split_path[len(split_path) - 3:]
        if full_path_as_key:
            key = os.path.join(family, person, pic)
        else:
            key = pic
        features[key] = feat_vec
        
    print('Calculation time elapsed: ' + str(int(time.time() - start)) + ' seconds.\n')
    
    return features, num_features


def save_feature_vectors(feature_vectors, num_features, filename):

    print('Saving feature vectors to file...\n')

    with open(filename, 'w', newline='') as csvfile:
        
        w = csv.writer(csvfile)
        w.writerow(['ImageFileName'] + ['Feature'+str(i) for i in range(num_features)])
        
        for key in feature_vectors.keys():
            w.writerow([key] + feature_vectors[key].tolist())


# Borrowed these from http://www.deepideas.net/unbalanced-classes-machine-learning/  # TODO
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

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
        metrics=[sensitivity, specificity]  # metrics=['accuracy']  # Imbalanced data set, so accuracy is not the best metric.
    )

    return model


def train_model(model, feature_dict, num_features, relation_dict):

    '''
    Training plan (76613631 pairs for 12379 images total):
    1. Select 100000 (random?) pairs of images
        - how do I ensure that each batch has the same ratio of pos/neg examples? - don't worry about that for now
    2. Create data and labels for these
    3. Call model.fit to train on these 100000 pairs
    4. Repeat steps 1-3 767 times to train incrementally for all 76 million+ image pairs
    '''

    print('Training model...\n')
    all_combinations = list(combinations(feature_dict.keys(), 2))  # Generate all pairwise image combinations.
    pairs_per_iteration = 100000
    num_combinations = len(all_combinations)
    num_iterations = (num_combinations // pairs_per_iteration) + 1
    sum_kin_relations = 0
    class_weights = {  # TODO : These weights are too strong toward '1'. No weights is too strong toward '0'.
        0:1,
        1:76000000/225000
    }

    # TODO need to make each iteration have about the same number of kin relationships.

    start = time.time()

    for iteration in range(num_iterations):  # Num combinations / num per iteration.
        
        print('\tIteration ' + str(iteration+1) + '/' + str(num_iterations))
        
        # Get the selection of images for this iteration.
        current_pairs = all_combinations[0:pairs_per_iteration]
        current_pair_count = len(current_pairs)
        del all_combinations[0:pairs_per_iteration]  # Free up memory as we go.

        # Create the list of relatives, and the difference vector list.
        diff_vectors = np.ndarray((current_pair_count, num_features))
        relations = np.ndarray((current_pair_count))
        num_kin_relations = 0
        for i, pair in enumerate(current_pairs):
            diff_vectors[i] = feature_dict[pair[0]] - feature_dict[pair[1]]
            person_0 = os.path.dirname(pair[0])
            person_1 = os.path.dirname(pair[1])
            relations[i] = 0
            if person_1 in relation_dict.get(person_0, []) or person_0 == person_1:
                num_kin_relations += 1
                relations[i] = 1
        relations = to_categorical(relations, num_categories)
        sum_kin_relations += num_kin_relations
        print('\tNumber of kin relation pairs in iteration: ' + str(num_kin_relations) + '/' + str(current_pair_count))
        
        # Run training.
        model.fit(
            diff_vectors,
            relations,
            batch_size = 1000,
            epochs = 1,
            validation_split = 0.2,
            class_weight = class_weights
        )
    
    end = int(time.time() - start)
    print('\nTraining time elapsed: ' + str(end) + ' seconds.')
    print('Training time per iteration: ' + str(end / num_iterations) + ' seconds.')
    print('Total number of kin relations: ' + str(sum_kin_relations) + '/' + str(num_combinations))

    return model


def save_model(model, filename):

    print('Saving model to file...\n')

    if not os.path.exists(filename):
        model.save(filename)


def make_predictions(model, feature_vectors, image_pairs):
    
    print('Making predictions...\n')
    num_pairs = len(image_pairs)
    preds = {
        'img_pair':[],
        'is_related':[]
    }
    
    for i in range(num_pairs):
        image_1, image_2 = image_pairs[i].split('-')
        diff_vector = feature_vectors[image_1] - feature_vectors[image_2]
        prediction = model.predict(diff_vector.reshape((1, len(diff_vector))))  # Returns vector of probabilities.
        preds['img_pair'].append(image_pairs[i])
        preds['is_related'].append(np.argmax(prediction, axis=1)[0])
    
    return pd.DataFrame(preds)


def save_predictions(preds):
    
    print('Saving predictions...\n')

    preds.to_csv(output_file, index = False)


if __name__ == '__main__':
    
    '''
    Initial plan:
    1. Use pre-trained face model - keras_vggface
    2. Iterate through all training faces to create their feature vectors (save these)
    3. Create smaller NN where the inputs are the raw differences of the feature vectors
    4. Train NN with ALL facial combinations, noting which are actually related
        -there are ~76 million combinations...do I need all? (Shia LaBeouf it)
        -what if I just use one from each person instead of all photos? (~3.8 million combinations)
    5. Run all test images through CNN to generate their feature vectors
    6. Run all test feature vectors/distances through NN model to predict kinship
    '''

    # Suppress some tensorflow INFO, WARNING, and ERROR messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load pre-trained facial model. Step 1.
    vggface = VGGFace()  # Uses default vgg16 model.
    print()

    # Get and prep training images and create feature vectors. Step 2.
    train_relationships = read_csv_file(train_relationships_file)
    train_rels_dict = defaultdict(list)
    print('Gathering kin relationships...\n')
    for i, row in train_relationships.iterrows():  # Store kin relationships for quick reference.
        key = row['p1'].replace('/', os.sep)
        value = row['p2'].replace('/', os.sep)
        train_rels_dict[key].append(value)
        train_rels_dict[value].append(key)

    # Check to see if the feature vectors have already been calculated.
    feature_vector_file = os.path.join(dataset_dir, 'VGGFace_feature_vectors_training.csv')
    feature_vectors = read_csv_file(feature_vector_file)
    num_features = -1
    if feature_vectors is None:
        print('Training feature vector file not found.\n')
        training_image_paths = []
        for folder in os.walk(train_files):  # Get all training image paths.
            training_image_paths.extend([os.path.join(folder[0], file) for file in folder[2]])
        feature_vectors, num_features = get_feature_vectors(vggface, training_image_paths, True)
        save_feature_vectors(feature_vectors, num_features, feature_vector_file)
        del training_image_paths
    else:  # Convert to dict.
        print('Reading stored training feature vectors...\n')
        labels = feature_vectors['ImageFileName']
        num_features = len(feature_vectors.columns) - 1
        temp_vectors = feature_vectors.drop('ImageFileName', axis=1).to_numpy()
        feature_vectors = {labels[i]: temp_vectors[i] for i in range(len(temp_vectors))}
        del temp_vectors, labels
    
    # Create comparison model. Step 3.
    comp_model = read_model(model_file)
    if comp_model is None:
        
        # Train comparison model. Step 4.
        comp_model = prep_model(num_features)
        comp_model = train_model(comp_model, feature_vectors, num_features, train_rels_dict)
        del feature_vectors
        del train_rels_dict
        save_model(comp_model, model_file)

    # Get and prep test images and create feature vectors. Step 5.
    feature_vector_file = os.path.join(dataset_dir, 'VGGFace_feature_vectors_test.csv')
    feature_vectors = read_csv_file(feature_vector_file)
    if feature_vectors is None:
        print('Test feature vector file not found.\n')
        test_image_paths = []
        for folder in os.walk(test_files):  # Get all training image paths.
            test_image_paths.extend([os.path.join(folder[0], file) for file in folder[2]])
        feature_vectors, num_features = get_feature_vectors(vggface, test_image_paths, False)
        save_feature_vectors(feature_vectors, num_features, feature_vector_file)
        del test_image_paths
    else:
        print('Reading stored test feature vectors...\n')
        labels = feature_vectors['ImageFileName']
        num_features = len(feature_vectors.columns) - 1
        temp_vectors = feature_vectors.drop('ImageFileName', axis=1).to_numpy()
        feature_vectors = {labels[i]: temp_vectors[i] for i in range(len(temp_vectors))}
        del temp_vectors, labels

    # Make predictions. Step 6.
    pairs = read_csv_file(submission_pairs_file)['img_pair']
    preds = make_predictions(comp_model, feature_vectors, pairs)
    save_predictions(preds)
