import tensorflow as tf
import numpy as np
from PIL import Image
import os
from data_augmentation import *
import random

DATA_NP_DIR = '/Users/ishariwaduwara-jayabahu/Desktop/Code/Python_Code/Breast_Cancer_AI/data/histology_image_full_dataset/' ##include trailing slash here
DATA_IMAGE_DIR = '/Users/ishariwaduwara-jayabahu/Desktop/Code/Python_Code/Breast_Cancer_AI/data/IDC_regular_ps50_idx5' ## do not include trailing slash here

def unison_shuffle_dataset(a, b):
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

def save_pngs_to_np_dataset(test_split=0.1, validation_split=0.1, test_patient_ids = [9075, 9176, 9225, 9382, 10299]):
    ##loads the data from Janowczyk's paper containing png files
    ##converts them to numpy arrays
    ##undersamples so there are equal numbers of idc+ and idc- images
    ##creates balanced x and y data from these arrays
    ##splits data into train/val data and test data
    ##saves these numpy arrays to the fulder under DATA_NP_DIR
    print("\n\nLoading Images ... Will take some time ........")
    DATA_DIR = DATA_IMAGE_DIR
    idc_positive = []
    idc_negative = []

    test_slide_positives = []
    test_slide_negatives = []

    for patient_id_folder in os.listdir(DATA_DIR):
        patient_path = os.path.join(DATA_DIR, patient_id_folder)
        
        if os.path.isdir(patient_path):
            patient_id = int(patient_id_folder)


            for class_folder in ['0', '1']:
                class_path = os.path.join(patient_path, class_folder)

                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            img = Image.open(img_path).convert('RGB')  # Ensure image is RGB
                            img = img.resize((50, 50))  # Ensure image size is 50x50
                            img_array = np.array(img)
                            
                            if img_array.shape == (50, 50, 3):

                                if class_folder == '1':
                                    if patient_id in test_patient_ids:
                                        test_slide_positives.append(img_array)
                                    else:
                                        idc_positive.append(img_array)
                                else:
                                    if patient_id in test_patient_ids:
                                        test_slide_negatives.append(img_array)
                                    else:
                                        idc_negative.append(img_array)

                            else:
                                print(f"Skipping image with incorrect shape: {img_path}")

                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

    # Convert lists to numpy arrays
    idc_negative = np.array(idc_negative) ##(193163, 50, 50, 3)
    idc_positive = np.array(idc_positive) ##(76234, 50, 50, 3)


    test_slide_negatives = np.array(test_slide_negatives) #(5575, 50, 50, 3)
    test_slide_positives = np.array(test_slide_positives) #(2552, 50, 50, 3) 
    # ^ together = 5575 + 2552 = 8127 test patches so far (we will add more test patches)


    # UNdersampling the idc_negative images from the patches that aren't already separated for the test slides
    # so we will undersample the negative images to also have only 76234 images
    num_positives = idc_positive.shape[0] 

    # Randomly select a subset of the IDC negative samples to match the number of IDC positive samples
    idx = np.random.choice(idc_negative.shape[0], num_positives, replace=False)
    idc_negative_undersampled = idc_negative[idx]

    print("idc_positive_shape:", idc_positive.shape)
    print("idc_negative_undersampled shape:", idc_negative_undersampled.shape)
    print("test_slide_positives:", test_slide_positives.shape)
    print("test_slide_negatives shape:", test_slide_negatives.shape)
    
    # Combine the undersampled IDC negative samples with the IDC positive samples
    x_data_balanced = np.concatenate([idc_negative_undersampled, idc_positive], axis=0)
    y_data_balanced = np.concatenate([np.zeros(num_positives), np.ones(num_positives)], axis=0)

    x_test_slide_data = np.concatenate([test_slide_negatives, test_slide_positives], axis=0)
    y_test_slide_data = np.concatenate([np.zeros(test_slide_negatives.shape[0]), np.ones(test_slide_positives.shape[0])], axis=0)

    x_data_balanced, y_data_balanced = unison_shuffle_dataset(x_data_balanced, y_data_balanced) 

    ##separate the data
    num_items = x_data_balanced.shape[0]
    num_test_slide_items = x_test_slide_data.shape[0]

    total_num_test_items = int(num_items*test_split)
    num_new_test_items = total_num_test_items - num_test_slide_items



    x_test_new = x_data_balanced[:num_new_test_items]
    y_test_new = y_data_balanced[:num_new_test_items]
    x_test = np.concatenate([x_test_slide_data, x_test_new], axis=0)
    y_test = np.concatenate([y_test_slide_data, y_test_new], axis=0)

    x_train_val = x_data_balanced[num_new_test_items:]
    y_train_val = y_data_balanced[num_new_test_items:]

    ##shuffle the new separated datasets again
    x_train_val, y_train_val = unison_shuffle_dataset(x_train_val, y_train_val) 
    x_test, y_test = unison_shuffle_dataset(x_test, y_test) 

    num_validation_items = int(num_items*validation_split)

    x_validation = x_train_val[:num_validation_items]
    y_validation = y_train_val[:num_validation_items]

    x_train_not_upsampled = x_train_val[num_validation_items:]
    y_train_not_upsampled = y_train_val[num_validation_items:]

    ##save the data
    np.save(DATA_NP_DIR+'X_TEST.npy', x_test)
    np.save(DATA_NP_DIR+'Y_TEST.npy', y_test)
    np.save(DATA_NP_DIR+'X_TRAIN_NOT_UPSAMPLED.npy', x_train_not_upsampled)
    np.save(DATA_NP_DIR+'Y_TRAIN_NOT_UPSAMPLED.npy', y_train_not_upsampled)
    np.save(DATA_NP_DIR+'X_VALIDATION.npy', x_validation)
    np.save(DATA_NP_DIR+'Y_VALIDATION.npy', y_validation)

    print("Data transferred from original pngs to np arrays and saved under:", DATA_NP_DIR)
    return

def load_data(batch_size=32, show_data_stats=False, augment_data = False, num_aug_visualizations=0,):
    
    ########################################## LOAD FILES ##########################################
    print("Loading Files ...")
    
    DATA_DIR = DATA_NP_DIR

    x_test = np.load(DATA_DIR + 'X_TEST.npy') ##test images
    y_test = np.load(DATA_DIR + 'Y_TEST.npy') ##test labels

    x_validation = np.load(DATA_DIR + 'X_VALIDATION.npy')
    y_validation = np.load(DATA_DIR + 'Y_VALIDATION.npy')

    x_train = np.load(DATA_DIR + 'X_TRAIN_NOT_UPSAMPLED.npy')
    y_train = np.load(DATA_DIR + 'Y_TRAIN_NOT_UPSAMPLED.npy')


    print("Finished Loading Files")     

    ########################################## NORMALIZE DATA ##########################################
    print("Normalizing Data...")
    x_test = x_test.astype('float32') / 256.0
    x_validation = x_validation.astype('float32') / 256.0
    x_train = x_train.astype('float32') / 256.0
    print("Finished Normalizing Data")

    ################################## DESCRIBE & ONE HOT ENCODE #########################################
    ##describe the data to see if everything is loaded correctly
    if show_data_stats:
        print('\n', '-'*10 + 'TRAINING DATA' + '-'*10)
        describe_data(x_train, y_train)

        print('\n', '-'*10 + 'VALIDATION DATA' + '-'*10)
        describe_data(x_validation, y_validation)

        print('\n', '-'*10 + 'TESTING DATA' + '-'*10)
        describe_data(x_test, y_test)


    # for i in range(60):
    #     n = random.randint(0,100000)
    #     if y_train[n] == 1:
    #         title = f"IDC+_{i}.png"
    #     else:
    #         title = f"IDC-_{i}.png"
    #     plt.imsave(title, (x_train[n] * 255).astype(np.uint8))


    ##one hot encode y data
    y_train = tf.keras.utils.to_categorical(y_train)
    y_validation = tf.keras.utils.to_categorical(y_validation)
    y_test = tf.keras.utils.to_categorical(y_test)

    ########################################## AUGMENT DATA ##########################################

    ##augment data if applicable
    if augment_data:
        train_ds = dataset_with_augmentation(x_train, y_train, batch_size=batch_size)
        for i in range(num_aug_visualizations):
            save_sample_augmentations(augmentation_layer=augmentation_layer(), sample_image=x_train[i], n = 8, id = i)
    else:
        train_ds = dataset_no_augmentation(x_train, y_train, batch_size=batch_size)

    validation_ds = dataset_no_augmentation(x_validation, y_validation, batch_size=batch_size)
    

    ########################################## RETURN ###############################################
    print('\n' + '-'*15 + 'DATA LOADED SUCCESSFULLY' + '-'*15 + "\n\n")

    return train_ds, validation_ds, x_test, y_test