import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


################################################### AUGMENTATION FUNCTIONS ###################################################
def dataset_no_augmentation(x_data, y_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def dataset_with_augmentation(x_data, y_data, batch_size, use_rotate=False):
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    augmentation = augmentation_layer(use_rotate=use_rotate)
    dataset = dataset.map(lambda x, y: (augmentation(x), y))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset 

def visualize_augmentation_tf(augmentation_layer, x_train, n=0):
    # Take one sample image from the training set
    sample_image = x_train[n]  # Assuming x_train is an array of images
    
    # Convert the sample image to a batch of one image
    sample_image = tf.expand_dims(sample_image, 0)

    # Plot original and augmented images
    plt.figure(figsize=(10, 10))

    # Display the original image
    plt.subplot(3, 3, 1)
    plt.title("Original Image")
    plt.imshow(sample_image[0])
    
    # Display 8 augmented images
    for i in range(8):
        augmented_image = augmentation_layer(sample_image)
        plt.subplot(3, 3, i+2)
        plt.imshow(augmented_image[0])
        # plt.axis('off')

    plt.show()
    input()
    visualize_augmentation_tf(augmentation_layer, x_train, n=n+1)

def augmentation_layer(use_rotate = False):
    if use_rotate:
        return augmentation_layer_rotate()
    else:
        return augmentation_layer_no_rotate()

def augmentation_layer_rotate():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomFlip('vertical'),
        tf.keras.layers.RandomRotation(1),
        tf.keras.layers.RandomCrop(height=40, width=40),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        CustomContrast(contrast_factor=1.25) 
    ])

def augmentation_layer_no_rotate():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomFlip('vertical'),
        tf.keras.layers.RandomCrop(height=40, width=40),
        # tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    ])

class CustomContrast(tf.keras.layers.Layer):
    def __init__(self, contrast_factor=1.0, **kwargs):
        super(CustomContrast, self).__init__(**kwargs)
        self.contrast_factor = contrast_factor

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        contrasted = tf.add(tf.multiply(tf.subtract(inputs, mean), self.contrast_factor), mean)
        return tf.clip_by_value(contrasted, 0.0, 1.0)  # Ensure pixel values are between 0 and 1

################################################### UPSAMPLING ROTATE FUNCTIONS ###################################################
def visualize_rotations(image, angles):
    # Visualize the rotations done by the upsampling given a sample image and the upsampling angles

    num_angles = len(angles)
    n_cols = 4  # Number of columns for the subplot grid
    n_rows = int(np.ceil((num_angles + 1) / n_cols))  # +1 for the original image

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))

    # Flatten the axs array for easy iteration
    axs = axs.ravel()

    # Show the original image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title('Original')
    axs[0].axis('off')

    # Show the rotated images in the remaining subplots
    for i, angle in enumerate(angles):
        rotated_image = rotate_image(image, angle)
        axs[i+1].imshow(rotated_image)
        axs[i+1].set_title(f'Rotated {angle}°')
        axs[i+1].axis('off')

    # Hide any unused subplots
    for j in range(num_angles + 1, n_rows * n_cols):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

def rotate_image(image, angle):
    """Rotates the image by the specified angle using Scipy."""
    # Rotate the image using scipy's rotate function
    rotated_image = rotate(image, angle, reshape=False, mode='nearest')
    return rotated_image

def rotate_images(images, labels, angles):
    # Rotates images by the specified angles and returns the upsampled dataset
    rotated_images = []
    rotated_labels = []

    for img, label in zip(images, labels):
        for angle in angles:
            rotated_img = rotate_image(img, angle)
            rotated_images.append(rotated_img)
            rotated_labels.append(label)

    return np.array(rotated_images), np.array(rotated_labels)