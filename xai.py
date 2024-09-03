import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

################################################# GRADCAM ##################################################

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

####################################################################################################
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
