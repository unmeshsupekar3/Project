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


    with open(data_path) as fp:
        data = json.load(fp)
        

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y


def plot_history(history):


    fig, axs = plt.subplots(2)

    
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy plot")

    
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error plot")

    plt.show()

def prepare_datasets(test_size, validation_size):

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):


    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):


    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    
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

    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # pick a sample to predict from the test set
    X_to_predict = X_test[100]
    y_to_predict = y_test[100]

    # predict sample
    predict(model, X_to_predict, y_to_predict)    

