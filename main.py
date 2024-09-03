from model import * 
from data import *
from xai import *

def main():
    ################################################### LOAD DATA ###########################################################
    global train_ds, validation_ds, x_test, y_test # this data is needed almost everywhere in the code

    # Load Data:
    # augmentation_mode should be in ["With Rotate", "No Rotate", "Any other string for no augmentation"]
    # ^ No rotate is often best since we already can upsample with rotation
    # upsampling_angles should contain all of the angles (deg) by which the original images should be rotated for the upsampled training data
    if input("\nPress enter to upsample data (may take some time). Input anything else to load raw data: ") == '':
        upsamples = [90, 180, 270]
        # upsamples = range(30, 360, 30)
    else:
        upsamples = []

    train_ds, validation_ds, x_test, y_test = load_data(
        validation_split=0.1,
        batch_size=32,
        show_data_stats=True,
        augmentation_mode = "No Rotate", 
        visualize_augmentation=False,
        upsampling_angles=upsamples,
        visualize_upsampling=False)
    
    ################################################### CREATE/LOAD MODEL ###################################################
    command = input("\nEnter file name or press enter to train new network: ")
    proceed_to_training = False


    if command!="": # some filename was entered
        model_name = command if '.' in command else command + ".keras"
        model = load_model(model_name, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6))

        # ask to train this model or not
        if input("\nPress enter to further train this model. Input anything else to proceed to testing: ") == '':
            ##the model has already been loaded in 'model'
            ##we just pass this model into train_model with its previous weights
            proceed_to_training = True

    else: # no filename was entered. make a new model from make_model and train it

        ##confirm before training model
        if input("\nPress enter to continue with training model. Input anything else to cancel: ") == '':
            model = make_model(show_summary=True, model_number=8)
            proceed_to_training = True

        else: 
            return # since the no model to load was provided, just return and quit the program


    ################################################### TRAIN MODEL ##########################################################
    if proceed_to_training:
        ##the appropriate model has been created and stored in 'model', either as a loaded model or as a new model from make_model
        model, model_filename = train_model(model, train_ds, validation_ds, epochs=5000, early_stopper_patience=-1)
        save_model(model, name = model_filename + ".keras")


    ################################################### TEST MODEL ############################################################
    test_model(x_test, y_test, model, prompt_user=True)


    ################################################### DISPLAY RESULTS/XAI ###################################################
    # cycle_results(x_test, y_test, model=model)
    # visualize_saliency_maps(model, x_test, y_test, num_images=20)
    visualize_gradcam(model, x_test, y_test, num_images=20)



if __name__ == "__main__":
   main()