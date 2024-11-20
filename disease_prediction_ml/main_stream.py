
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import statistics

class DiseasePredict:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.final_svm_model = SVC(probability=True)
        self.final_nb_model = GaussianNB()
        self.final_rf_model = RandomForestClassifier(random_state=18)
        self.data_dict = {}

    def read_data(self, data_path):
        data = pd.read_csv(data_path).dropna(axis=1)
        data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
        data['prognosis'] = data['prognosis'].str.strip().str.lower().replace(" ", "_")
        return data

    def encode(self, data):
        data["prognosis"] = self.encoder.fit_transform(data["prognosis"])
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
        symptoms = X.columns.values
        symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}
        self.data_dict = {"symptom_index": symptom_index, "predictions_classes": self.encoder.classes_}
        return X_train, X_test, y_train, y_test, X, y

    def cross_val(self, X, y):
        models = {
            "SVC": SVC(),
            "Gaussian NB": GaussianNB(),
            "Random Forest": RandomForestClassifier(random_state=18)
        }
        results = {}
        for model_name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            results[model_name] = np.mean(scores) * 100
        return results

    def train_classifiers(self, X_train, X_test, y_train, y_test):
        accuracy_results = {}
        
        self.final_svm_model.fit(X_train, y_train)
        svm_preds = self.final_svm_model.predict(X_test)
        accuracy_results['SVM'] = accuracy_score(y_test, svm_preds) * 100
        
        self.final_nb_model.fit(X_train, y_train)
        nb_preds = self.final_nb_model.predict(X_test)
        accuracy_results['Naive Bayes'] = accuracy_score(y_test, nb_preds) * 100
        
        self.final_rf_model.fit(X_train, y_train)
        rf_preds = self.final_rf_model.predict(X_test)
        accuracy_results['Random Forest'] = accuracy_score(y_test, rf_preds) * 100
        
        return accuracy_results

    def model_fit(self, X, y):
        self.final_svm_model.fit(X, y)
        self.final_nb_model.fit(X, y)
        self.final_rf_model.fit(X, y)

    def predictDisease(self, symptoms):
        symptoms = symptoms.split(",")
        input_data = [0] * len(self.data_dict["symptom_index"])
        for symptom in symptoms:
            symptom = symptom.strip().lower().replace(" ", "_")
            index = self.data_dict["symptom_index"].get(symptom, None)
            if index is not None:
                input_data[index] = 1

        input_data_df = pd.DataFrame([input_data], columns=list(self.data_dict["symptom_index"].keys()))
        rf_prediction = self.data_dict["predictions_classes"][self.final_rf_model.predict(input_data_df)[0]]
        nb_prediction = self.data_dict["predictions_classes"][self.final_nb_model.predict(input_data_df)[0]]
        svm_prediction = self.data_dict["predictions_classes"][self.final_svm_model.predict(input_data_df)[0]]
        final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])

        predictions = {
            "rf_model_prediction": rf_prediction,
            "naive_bayes_prediction": nb_prediction,
            "svm_model_prediction": svm_prediction,
            "final_prediction": final_prediction
        }
        return predictions

if __name__ == "__main__":
    st.title('Disease Predictor')
    st.markdown("""
    ## Welcome to the Disease Prediction Application!
    Please enter your symptoms below, separated by commas (e.g., Itching, Skin Rash).
    """)
    
    symptoms_input = st.text_input("Enter your symptoms (comma-separated):", placeholder="e.g., Itching, Skin Rash")

    if st.button("Predict"):
        if symptoms_input:
            with st.chat_message("user"):
                st.write(symptoms_input)

            training_path = r'C:\Users\unmes\Documents\Projects\Project\disease_prediction_ml\data\Training.csv'
            disease = DiseasePredict()
            data = disease.read_data(training_path)
            X_train, X_test, y_train, y_test, X, y = disease.encode(data)

            st.write("### Performing Cross-Validation...")
            cross_val_results = disease.cross_val(X, y)
            for model, accuracy in cross_val_results.items():
                st.write(f"{model} - Cross Validation Accuracy: {accuracy:.2f}%")

            st.write("### Training individual classifiers and evaluating on the test set...")
            accuracy_results = disease.train_classifiers(X_train, X_test, y_train, y_test)
            for model, accuracy in accuracy_results.items():
                st.write(f"{model} Classifier Test Accuracy: {accuracy:.2f}%")

            st.write("### Training final models on the full dataset...")
            disease.model_fit(X, y)

            predictions = disease.predictDisease(symptoms_input)
            with st.chat_message("bot"):
                st.write("### Predictions")
                st.write(f"🔹 Random Forest Prediction: {predictions['rf_model_prediction']}")
                st.write(f"🔹 Naive Bayes Prediction: {predictions['naive_bayes_prediction']}")
                st.write(f"🔹 SVM Prediction: {predictions['svm_model_prediction']}")
                st.write(f"🔹 **Final Prediction**: {predictions['final_prediction']}")
        else:
            st.warning("Please enter some symptoms.")
