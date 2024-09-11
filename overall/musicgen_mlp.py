import json
#import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
#from tensorflow.keras import layers



# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = r"C://Users//unmes//Downloads//musicgen//Data//data.json"

def load_data(data_path):
    """Load training dataset from json file.
        data path : Path to json file containing data
        X : Inputs
        y : Targets
    """

    with open(data_path) as fp:
        data = json.load(fp)
        

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y


def plot_history(history):
    """Plot accuracy/loss for training/validation set as a function of the epochs
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def predict(model, X, y):
   

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    # Mapping the data labels
    if(y==0):
        tar_val="blues"
    elif(y==1):
        tar_val="classical"
    elif(y==2):
        tar_val="country"
    elif(y==3):
        tar_val="disco"
    elif(y==4):
        tar_val="hiphop"
    elif(y==5):
        tar_val="jazz"
    elif(y==6):
        tar_val="metal"
    elif(y==7):
        tar_val="pop"
    elif(y==8):
        tar_val="reggae"
    elif(y==9):
        tar_val="rock"
        
    if(predicted_index==0):
        out_val="blues"
    elif(predicted_index==1):
        out_val="classical"
    elif(predicted_index==2):
        out_val="country"
    elif(predicted_index==3):
        out_val="disco"
    elif(predicted_index==4):
        out_val="hiphop"
    elif(predicted_index==5):
        out_val="jazz"
    elif(predicted_index==6):
        out_val="metal"
    elif(predicted_index==7):
        out_val="pop"
    elif(predicted_index==8):
        out_val="reggae"
    elif(predicted_index==9):
        out_val="rock" 

    print("Target: {}, Predicted label: {}, Target_value:{}, Output_value:{} ".format(y, predicted_index, tar_val, out_val))




if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)




 # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st Hidden layer         
        keras.layers.Dense(512, activation ='relu', kernel_regularizer=keras.regularizers.l2(0.1)),
        keras.layers.Dropout(0.2),

        # 2nd Hidden layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer= keras.regularizers.l2(0.3)),
        keras.layers.Dropout(0.3),

        # 3rd Hidden layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer= keras.regularizers.l2(0.2)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)
    plot_history(history)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
  
    
    X_to_predict = X_test[100]
    y_to_predict = y_test[100]

    # predict sample
    predict(model, X_to_predict, y_to_predict)   




