import os
import numpy as np
import librosa
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to GTZAN dataset
DATA_PATH = r'C:\Users\unmes\Documents\Projects\music_genre\data\inputs\genres_original' 

# Genres
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def prepare_data(data_path):
    features = []
    labels = []

    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            mfcc = extract_features(file_path)
            features.append(mfcc)
            labels.append(genres.index(genre))

    X = np.array(features)
    y = to_categorical(np.array(labels), num_classes=len(genres))
    return X, y

def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_genre(model, file_path):
    mfcc = extract_features(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre

def initialize():
    if not os.path.exists('music_genre_classification_model.h5'):
        X, y = prepare_data(DATA_PATH)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = build_model((X_train.shape[1],), len(genres))
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
        model.save('music_genre_classification_model.h5')
    else:
        model = load_model('music_genre_classification_model.h5')
    
    return model

def main(model):
    st.title("Music Genre Classification")
    st.write("Upload an audio file to classify its genre")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(uploaded_file, format='audio/wav')

        if st.button("Classify"):
            genre = predict_genre(model, "temp_file")
            st.write(f"The predicted genre is: {genre}")

if __name__ == "__main__":
    model = initialize()
    main(model)
