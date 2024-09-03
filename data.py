import tensorflow as tf
import numpy as np

from data_aug_upsample import *

DATA_DIR = '/Users/ishariwaduwara-jayabahu/Desktop/Code/Python_Code/Breast_Cancer_AI/histology_image_dataset/'

def unisonShuffleDataset(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def describe_data(a, b):
    print('Total number of images (x_data):', len(a))
    print('Total number of images (y_data):', len(a))
    print('Number of IDC(-) Images:', np.sum(b==0))
    print('Number of IDC(+) Images:', np.sum(b==1))
    print('Percentage of positive images:', str(round(100*np.mean(b), 2)) + '%') 
    print('Image shape (Width, Height, Channels):', a[0].shape)
    print()

def split_test_data(test_split=0.1):
    ##this is only run once to separate testing data from train/validation data
    ##It saves the test data in a different file so it cannot be touched when training

    x_all = np.load(DATA_DIR + 'X.npy') # images
    y_all = np.load(DATA_DIR + 'Y.npy') # labels associated to images (0 = no IDC, 1 = IDC)

    x_all, y_all = unisonShuffleDataset(x_all, y_all) ##shuffle the data around before 

    ##separate the data
    num_items = len(x_all)
    num_test_items = int(num_items*test_split)

    x_test = x_all[:num_test_items]
    y_test = y_all[:num_test_items]

    x_train_val = x_all[num_test_items:]
    y_train_val = y_all[num_test_items:]

    ##shuffle the new separated datasets again
    x_train_val, y_train_val = unisonShuffleDataset(x_train_val, y_train_val) 
    x_test, y_test = unisonShuffleDataset(x_test, y_test) 

    ##save the data
    np.save(DATA_DIR+'X_TEST.npy', x_test)
    np.save(DATA_DIR+'Y_TEST.npy', y_test)
    np.save(DATA_DIR+'X_TRAIN_VAL.npy', x_train_val)
    np.save(DATA_DIR+'Y_TRAIN_VAL.npy', y_train_val)

def load_data(validation_split=0.1, batch_size=32, show_data_stats=False,
              augmentation_mode="", visualize_augmentation=False,
              upsampling_angles=[], visualize_upsampling=False):
    
    ########################################## LOAD FILES ##########################################
    # x_all = np.load(DATA_DIR + 'X.npy') # all images
    y_all = np.load(DATA_DIR + 'Y.npy') # all labels associated to images (0 = no IDC, 1 = IDC)
    num_items = len(y_all)

    x_test = np.load(DATA_DIR + 'X_TEST.npy') ##test images
    y_test = np.load(DATA_DIR + 'Y_TEST.npy') ##test labels

    x_train_val = np.load(DATA_DIR + 'X_TRAIN_VAL.npy') ##train/val images
    y_train_val = np.load(DATA_DIR + 'Y_TRAIN_VAL.npy') ##train/val images

    ########################################## PROCESS DATA ##########################################
    ##shuffle the loaded data
    x_train_val, y_train_val = unisonShuffleDataset(x_train_val, y_train_val) 
    x_test, y_test = unisonShuffleDataset(x_test, y_test) 

    ##normalize both sets of data to be between 0 and 1 instead of 0 and 255
    x_test = x_test.astype('float32') / 256.0
    x_train_val = x_train_val.astype('float32') / 256.0

    ########################################## SPLIT DATA ##########################################
    ##split the train/val data into either train or validation 
    num_validation_items = int(num_items*validation_split)

    x_validation = x_train_val[:num_validation_items]
    y_validation = y_train_val[:num_validation_items]

    x_train_not_upsampled = x_train_val[num_validation_items:]
    y_train_not_upsampled = y_train_val[num_validation_items:]

    ########################################## UPSAMPLING ##########################################
    ##upsample the training data into numpy arrays called x_train, y_train
    if upsampling_angles != []:
        if visualize_upsampling:
            visualize_rotations(x_train_not_upsampled[0], upsampling_angles)
        
        # Upsample the training data
        x_train_rotated, y_train_rotated = rotate_images(x_train_not_upsampled, y_train_not_upsampled, upsampling_angles)
        
        # Combine the original and rotated images
        x_train = np.concatenate([x_train_not_upsampled, x_train_rotated], axis=0)
        y_train = np.concatenate([y_train_not_upsampled, y_train_rotated], axis=0)
    else:
        x_train = x_train_not_upsampled
        y_train = y_train_not_upsampled

    ##shuffle individual data sets again
    x_train, y_train = unisonShuffleDataset(x_train, y_train)
    x_validation, y_validation = unisonShuffleDataset(x_validation, y_validation)

    ################################## DESCRIBE/ONE HOT ENCODE ##########################################
    ##describe the data to see if everything is loaded correctly
    if show_data_stats:
        print('\n', '-'*10 + 'TRAINING DATA' + '-'*10)
        describe_data(x_train, y_train)

        print('\n', '-'*10 + 'VALIDATION DATA' + '-'*10)
        describe_data(x_validation, y_validation)

        print('\n', '-'*10 + 'TESTING DATA' + '-'*10)
        describe_data(x_test, y_test)

    ##one hot encode y data
    y_train = tf.keras.utils.to_categorical(y_train)
    y_validation = tf.keras.utils.to_categorical(y_validation)
    y_test = tf.keras.utils.to_categorical(y_test)

    ########################################## AUGMENT DATA ##########################################
    ##augment data if applicable
    if augmentation_mode == "With Rotate":
        train_ds = dataset_with_augmentation(x_train, y_train, batch_size=batch_size, use_rotate=True)
        if visualize_augmentation:
            visualize_augmentation_tf(augmentation_layer(use_rotate=True), x_train)
    elif augmentation_mode == "No Rotate":
        train_ds = dataset_with_augmentation(x_train, y_train, batch_size=batch_size, use_rotate=False)
        if visualize_augmentation:
            visualize_augmentation_tf(augmentation_layer(use_rotate=False), x_train)
    else:
        train_ds = dataset_no_augmentation(x_train, y_train, batch_size=batch_size)

    validation_ds = dataset_no_augmentation(x_validation, y_validation, batch_size=batch_size)
    

    ########################################## RETURN ###############################################
    print('\n' + '-'*15 + 'DATA LOADED SUCCESSFULLY' + '-' * 15 + "\n\n")

    return train_ds, validation_ds, x_test, y_test