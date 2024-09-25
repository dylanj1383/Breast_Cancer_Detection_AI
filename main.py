from model import * 
from data import *
from xai import *
from image_processing import *

ask_to_further_train_model = False ##doesn't further train model by default
ask_to_test_model = False ## tests model by default
command = "m2_checkpoints/m2_checkpoint_0028.keras" # set to None to prompt user. Set to empty string to train new model. Set to filename to load new model.

def main():
    ################################################### LOAD DATA ###########################################################
    global command, ask_to_test_model, ask_to_further_train_model
    global train_ds, validation_ds, x_test, y_test # this data is needed almost everywhere in the code

    # Load Data:
    train_ds, validation_ds, x_test, y_test = load_data(
        batch_size=32,
        show_data_stats=True,
        augment_data=True,
        num_aug_visualizations=0,)
    
    ################################################### CREATE/LOAD MODEL ###################################################
    if command == None:
        command = input("\nEnter file name or press enter to train new network: ")
    

    proceed_to_training = False


    if command!="": # some filename was entered
        model_name = command if '.' in command else command + ".keras"
        model = load_model(model_name, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6))

        # ask to train this model or not
        if ask_to_further_train_model and input("\nPress enter to further train this model. Input anything else to proceed to testing: ") == '':
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
        model, model_filename = train_model(model, train_ds, validation_ds, epochs=50, early_stopper_patience=-1)
        save_model(model, name = model_filename + ".keras")


    ################################################### TEST MODEL ############################################################
    test_model(x_test, y_test, model, prompt_user=ask_to_test_model)


    ################################################### DISPLAY RESULTS/XAI ###################################################
    # cycle_results(x_test, y_test, model=model)
    # visualize_saliency_maps(model, x_test, y_test, num_images=40)



if __name__ == "__main__":
    # save_pngs_to_np_dataset()
    # save_full_patient_slides()


    main()


    # model = load_model(command, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6))
    # # save_patched_heatmaps(model)
    # save_sliding_heatmaps(model, ids = [9075, 9176, 9225, 9382, 10299])
    