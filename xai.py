import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

############################################### GRADCAM ###############################################

def compute_gradcam(model, img_tensor, class_index, last_conv_layer_name='conv2d_2', last_layer_name='dense_1'):
    # Get the gradients of the predicted class with respect to the output of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.get_layer(last_layer_name).output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    # Compute the gradients of the loss with respect to the last conv layer output
    grads = tape.gradient(loss, conv_outputs)

    # Compute the guided gradients (ReLU on the gradients)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute the Grad-CAM by performing a weighted sum of the guided gradients and conv outputs
    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[None, None, None, :]

    gradcam = tf.reduce_sum(tf.multiply(guided_grads, conv_outputs), axis=-1).numpy()

    # Apply ReLU and normalize the Grad-CAM output
    gradcam = np.maximum(gradcam, 0)
    gradcam = gradcam / np.max(gradcam)

    return gradcam

def visualize_gradcam(model, x_test, y_test, num_images=5, 
                      last_conv_layer_name='conv2d_2', last_layer_name = 'dense_1', 
                      img_dimensions=(50, 50, 3)):
    
    for idx in range(num_images):
        img = x_test[idx]
        label = np.argmax(y_test[idx])  # True label
        img_tensor = tf.convert_to_tensor(img[None, ...])

        # Get model predictions and predicted label
        predictions = model(img_tensor)
        predicted_label = np.argmax(predictions[0])

        # Compute Grad-CAM
        gradcam = compute_gradcam(model, img_tensor, predicted_label, 
                                  last_conv_layer_name=last_conv_layer_name, 
                                  last_layer_name=last_layer_name)

        # Resize the Grad-CAM to the size of the input image
        gradcam = tf.image.resize(gradcam[..., tf.newaxis], img_dimensions[:2]).numpy()

        # Plotting the original image, Grad-CAM heatmap, and overlay
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img.reshape(img_dimensions))
        axs[0].set_title(f"Original Image\nLabel: {label}, Prediction: {predicted_label}")
        axs[0].axis('off')

        axs[1].imshow(gradcam.squeeze(), cmap='hot')
        axs[1].set_title("Grad-CAM Heatmap")
        axs[1].axis('off')

        axs[2].imshow(img.reshape(img_dimensions))
        axs[2].imshow(gradcam.squeeze(), cmap='jet', alpha=0.5)
        axs[2].set_title("Overlay")
        axs[2].axis('off')

        plt.show()

        plt.close()

############################################### SALIENCY MAP ###############################################
def visualize_saliency_maps(model, x_test, y_test, num_images, img_dimensions=(50, 50, 3)):
    for idx in range(num_images):
        img = x_test[idx]
        label = np.argmax(y_test[idx])  # True label
        img_tensor = tf.convert_to_tensor(img[None, ...])

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            predicted_label = np.argmax(predictions[0])  # Model's prediction
            loss = predictions[:, predicted_label]

        grads = tape.gradient(loss, img_tensor)
        saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()

        # Normalize the saliency map for better visualization
        saliency = saliency[0]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        # Plotting the original image, saliency map, and overlay
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img.reshape(img_dimensions))
        axs[0].set_title(f"Original Image\nLabel: {label}, Prediction: {predicted_label}")
        axs[0].axis('off')

        axs[1].imshow(saliency, cmap='hot')
        axs[1].set_title("Saliency Map")
        axs[1].axis('off')

        axs[2].imshow(img.reshape(img_dimensions))
        axs[2].imshow(saliency, cmap='jet', alpha=0.35)
        axs[2].set_title("Overlay")
        axs[2].axis('off')

        
        plt.show()

        plt.close()


def cycle_results(x_data, y_data, dimensions=(50, 50, 3), model=""): 
    ##if no model has been provided, show the data and its labels only
    if model == "":
        output_data = np.array([])

    ##if a model has been provided, show the data, labels, and model predictions
    else:
        output_data = model.predict(x_data, verbose=0)

    ##cycle the results
    n = 0

    while True:
        input_image = x_data[n].reshape(dimensions)

        # Resize the image to twice its size
        resized_image = cv2.resize(input_image, (dimensions[1] * 4, dimensions[0] * 4))

        title = "Label: " + str(np.argmax(y_data[n])) + " "

        if output_data.size > 0:
            title = title + "Prediction: " + str(np.argmax(output_data[n]))

        # Display the resized image
        cv2.imshow("Image", resized_image)

        print(title, end="\r")

        ##if "back" or any similar keys are pressed, show the previous example
        if cv2.waitKey(0) in [ord("b"), ord("<"), ord(","), ord("-")]:
            n -= 1
            if n < 0:
                n = len(x_data) - 1  # Loop back to the last image if at the start

        elif cv2.waitKey(0) == ord('q'):
            break

        ##otherwise, show the next example when a key is pressed
        else:
            n += 1
            if n >= len(x_data):
                n = 0  # Loop back to the first image if at the end
######################################################################################