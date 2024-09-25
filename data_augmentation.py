import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

################################################### AUGMENTATION FUNCTIONS ###################################################
def dataset_no_augmentation(x_data, y_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def dataset_with_augmentation(x_data, y_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    augmentation = augmentation_layer()
    dataset = dataset.map(lambda x, y: (augmentation(x), y))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset 


def save_sample_augmentations(augmentation_layer, sample_image, n=5, id=0):
    """
    Save the original image and n augmented versions of the image to files.
    
    Parameters:
    - augmentation_layer: A function or layer that applies augmentation to images
    - sample_image: The original image to be augmented (as a numpy array or tensor)
    - n: The number of augmented images to save (default: 5)
    """
    # Check if the sample image is a TensorFlow tensor, if not, assume it's a NumPy array
    if isinstance(sample_image, np.ndarray):
        # If the array is not in float format, normalize it to [0, 1]
        if sample_image.dtype != np.float32:
            sample_image = sample_image.astype(np.float32) / 255.0
    else:
        # Convert TensorFlow tensor to float and normalize
        if sample_image.dtype != tf.float32:
            sample_image = tf.cast(sample_image, tf.float32) / 255.0

    # Save the original image, converting back to [0, 255] for saving
    plt.imsave(f"Original_{id}.png", (sample_image * 255).astype(np.uint8))
    
    # Save n augmented images
    for i in range(1, n + 1):
        # Apply augmentation
        augmented_image = augmentation_layer(tf.expand_dims(sample_image, 0))
        
        # Convert the augmented image back to [0, 255] and save
        augmented_image_np = (augmented_image[0].numpy() * 255).astype(np.uint8)
        plt.imsave(f"OAugmentation_{id}_num_{i}.png", augmented_image_np)

    print(f"Saved {n} augmented images and the original image and augmentations with id {id}.")

def custom_random_rotation(image):
    # Randomly rotate the image by 0, 90, 180, or 270 degrees    
    return tf.image.rot90(image, k=random.randint(0, 3))

def augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),  # Random horizontal and vertical flips
        tf.keras.layers.Lambda(lambda x: tf.map_fn(custom_random_rotation, x, dtype=tf.float32))
    ])