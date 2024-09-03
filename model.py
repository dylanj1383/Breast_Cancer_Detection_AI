import tensorflow as tf
from tensorflow.keras import layers, models
import os

TENSORBOARD_DIR = '/Users/ishariwaduwara-jayabahu/Desktop/Code/Python_Code/Breast_Cancer_AI/tensorboard_logs/'
SAVED_MODELS_DIR = '/Users/ishariwaduwara-jayabahu/Desktop/Code/Python_Code/Breast_Cancer_AI/saved_models/'
    
def make_model(show_summary=True, model_number = 8):
    models = [make_model1, make_model2, make_model3, make_model4, make_model5, make_model6, make_model7, make_model8]
    return models[model_number-1](show_summary=show_summary)

def train_model(model, train_ds, validation_ds, epochs=1, early_stopper_patience = 20):
    train_callbacks = []

    if early_stopper_patience > 0:
        ##early stopper stops training if validation loss stops decreasing
        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                        min_delta=0, 
                                                        patience=early_stopper_patience, 
                                                        verbose=1,
                                                        restore_best_weights=True)
        
        train_callbacks.append(early_stopper)

    ##save data to tensorboard for visualization
    model_filename = input("Enter model name here: ")

    use_tensorboard = input("Do you want to log to tensorboard? Input any character if so: ")

    use_checkpoints = input("Do you want to save model checkpoints? Input any character if so: ")

    

    if use_tensorboard != "":
        path = os.path.join(TENSORBOARD_DIR, model_filename+"_tensorboard_data") 

        assert not os.path.exists(path)

        os.makedirs(path) 
        log_dir = TENSORBOARD_DIR+ model_filename+"_tensorboard_data"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                            histogram_freq=1, 
                                                            write_graph=True,
                                                            # update_freq='batch'
                                                            )
        train_callbacks.append(tensorboard_callback)

    if use_checkpoints != '':
        checkpoint_dir = os.path.join(SAVED_MODELS_DIR, model_filename + '_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, model_filename + '_checkpoint_{epoch:04d}.keras'),
            save_freq='epoch',
        )

        train_callbacks.append(checkpoint_callback)

    #fit model
    model.fit(train_ds,
            epochs=epochs, 
            verbose = 1,
            validation_data=validation_ds,
            callbacks=train_callbacks) 

    return model, model_filename

def save_model(model, name = ""):
    if name != "":    
        model.save(SAVED_MODELS_DIR + name)
        print("Model saved under filename:", name)
    else:
        print("No filename entered. Not saving model.")

def test_model(x_data, y_data, model, prompt_user=True):
    if prompt_user:
        command = input("Press enter to test this model. Input anything else to skip testing. ")
    else:
        command = ''
    
    
    if command == '':
        results = model.evaluate(x_data, y_data)
        print("testing loss:", round(results[0], 5), " "*6+"testing accuracy:", round(results[1], 5))
        return round(results[0], 5), round(results[1], 4)
    
def load_model(load_model_name, show_summary = True, optimizer = 'adam'):
    try:
        model = tf.keras.models.load_model(SAVED_MODELS_DIR + load_model_name)
        model.compile(optimizer=optimizer,
                loss='categorical_crossentropy', 
                metrics=['accuracy']) 
        if show_summary:
            model.summary()
        return model
    

    except:
        ##if user has inputted an invalid filename, ask them to try again
        print("Invalid filename. Enter a valid model saved in " + SAVED_MODELS_DIR)
        command = input("Enter a valid name for a saved model: ")
        if '.' in command:
            return load_model(command)
        else:
            return load_model(command + ".keras")
        
########################################################################################################################################
################################################## ALL MODEL VERSIONS BELOW ############################################################
########################################################################################################################################

def make_model1(show_summary = True):
    ##make model

    model=tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(6,(3,3),input_shape=(50,50,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(10,(3,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Flatten()) 

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']) 
    
    ##show summary if requested
    if show_summary:
        model.summary()

    return model

def make_model2(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model3(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model4(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.PReLU()) ##try prely activation instead of relu
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model5(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.PReLU()) ##try prelu activation instead of relu
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    # Use a lower initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Default is 0.001

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model6(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.PReLU()) ##try prelu activation instead of relu
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    # Use a lower initial learning rate
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=100000,  # Number of steps after which learning rate decays
        decay_rate=0.96,     # The factor by which the learning rate will decay
        staircase=True)      # Whether to apply decay in a discrete staircase function

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model7(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.PReLU()) ##try prelu activation instead of relu
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[6],  # After 6 epochs
        values=[0.0001, 0.00005]  # Start with 0.0001, then 0.00005 after 6 epochs
    )
    # Use a lower initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    if show_summary:
        model.summary()

    return model

def make_model8(show_summary=True):
    model = tf.keras.models.Sequential()

    # Add convolutional layers with increased filters and different activation functions
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(50, 50, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Dense layers with more units
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # Changed to softmax for multi-class classification

    # Use a lower initial learning rate with weight decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 'Precision', 'Recall'])

    if show_summary:
        model.summary()

    return model