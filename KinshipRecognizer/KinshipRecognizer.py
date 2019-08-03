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
import random
from itertools import combinations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
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
    num_features = -1
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


def merge_feature_vectors(feature_vectors, num_features):
    
    # feature_vectors is a dict where key=<path\filename> and value=<ndarray of features>
    
    # Get all vectors aggregated by person.
    averaged_vectors = {}
    for f in feature_vectors.keys():
        person = os.path.dirname(f)  # Gets <family>\<person>
        if person not in averaged_vectors:
            averaged_vectors[person] = []
        averaged_vectors[person].append(feature_vectors[f])
    
    # Average each vector by person.
    for p in averaged_vectors.keys():
    
        vect_sum = averaged_vectors[p].pop(0)  # Set equal to first vector.
        count = 1
        for feat_vec in averaged_vectors[p]:
            vect_sum += feat_vec
            count += 1
    
        averaged_vectors[p] = vect_sum / count
    
    filename = os.path.join(dataset_dir, 'VGGFace_avg_feature_vectors_training.csv')
    save_feature_vectors(averaged_vectors, num_features, filename)


def save_feature_vectors(feature_vectors, num_features, filename):

    print('Saving feature vectors to file...\n')

    with open(filename, 'w', newline='') as csvfile:
        
        w = csv.writer(csvfile)
        w.writerow(['ImageFileName'] + ['Feature'+str(i) for i in range(num_features)])
        
        for key in feature_vectors.keys():
            w.writerow([key] + feature_vectors[key].tolist())


def prep_model(num_features):
    
    print('Prepping model...\n')

    model = Sequential()

    model.add(Dense(
        100,
        activation='relu',
        input_shape=(num_features,)
    ))
    '''.add(Dense(
        32,
        activation='relu'
    ))'''
    '''model.add(Dense(
        32,
        activation='relu'
    ))'''
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


def train_model(model, feature_dict, num_features, kin_combinations):

    print('Training model...\n')

    all_combinations = list(combinations(feature_dict.keys(), 2))  # Generate all pairwise people combinations.
    non_kin_combinations = { (pair if pair not in kin_combinations else None):False for pair in all_combinations}
    del non_kin_combinations[None], all_combinations
    non_kin_combinations = list(non_kin_combinations)
    num_kin_relations = len(kin_combinations.keys())
    num_non_kin_relations = len(non_kin_combinations)
    pairs_per_iteration = num_kin_relations * 2
    num_iterations = (num_non_kin_relations // num_kin_relations) + 1

    start = time.time()
    for iteration in range(num_iterations):
        
        print('\tIteration ' + str(iteration + 1) + '/' + str(num_iterations))
        
        # Get the selection of images for this iteration.
        current_pairs = non_kin_combinations[0:pairs_per_iteration // 2]
        current_pairs.extend(list(kin_combinations.keys()))
        random.shuffle(current_pairs)
        del non_kin_combinations[0:pairs_per_iteration//2]  # Free up memory as we go.

        # Create the list of relatives, and the difference vector list.
        diff_vectors = []  # np.ndarray((current_pair_count, num_features))
        relations = []  # np.ndarray((current_pair_count))
        for pair in current_pairs:
            if pair[0] not in feature_dict or pair[1] not in feature_dict:
                continue
            diff_vectors.append(np.absolute(feature_dict[pair[0]] - feature_dict[pair[1]]) / 0.000001)  # Lazy feature scaling.
            if kin_combinations.get((pair[0], pair[1]), False) or kin_combinations.get((pair[1], pair[0]), False):
                relations.append(1)
            else:
                relations.append(0)
        diff_vectors = np.stack(diff_vectors)
        relations = to_categorical(relations, num_categories)

        class_weight = {
            0:1,
            1:(len(current_pairs)-num_kin_relations)/num_kin_relations
        }

        # Run training.
        model.fit(
            diff_vectors,
            relations,
            batch_size = 100,
            epochs = 10,
            class_weight=class_weight,
            validation_split = 0.1
        )
    
    total = int(time.time() - start)
    print('\nTraining time elapsed: ' + str(total) + ' seconds.')
    print('Training time per iteration: ' + str(total / num_iterations) + ' seconds.')

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
        diff_vector = np.absolute(feature_vectors[image_1] - feature_vectors[image_2]) / 0.000001  # Lazy feature scaling.
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
        -there are ~76 million combinations...how can I reduce this?
        -what if I just use one from each person instead of all photos?
        -what if I average the vectors from each person? (~2.6 million combinations)
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
    train_rels_dict = {}
    print('Gathering kin relationships...\n')
    for i, row in train_relationships.iterrows():  # Store kin relationships for quick reference.
        p1 = row['p1'].replace('/', os.sep)
        p2 = row['p2'].replace('/', os.sep)
        train_rels_dict[(p1, p2)] = True

    # Check to see if the feature vectors have already been calculated.
    feature_vector_file = os.path.join(dataset_dir, 'VGGFace_avg_feature_vectors_training.csv')
    feature_vectors = read_csv_file(feature_vector_file)  # TEMP
    num_features = -1
    if feature_vectors is None:
        print('Training feature vector file not found.\n')
        training_image_paths = []
        for folder in os.walk(train_files):  # Get all training image paths.
            training_image_paths.extend([os.path.join(folder[0], file) for file in folder[2]])
        feature_vectors, num_features = get_feature_vectors(vggface, training_image_paths, True)
        merge_feature_vectors(feature_vectors, num_features)
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
