import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
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
        for model_name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            print(f"{model_name} - Cross Validation Accuracy: {np.mean(scores) * 100:.2f}%")
    
    def train_classifiers(self, X_train, X_test, y_train, y_test):
        self.final_svm_model.fit(X_train, y_train)
        svm_preds = self.final_svm_model.predict(X_test)
        print(f"SVM Classifier Test Accuracy: {accuracy_score(y_test, svm_preds) * 100:.2f}%")
        
        self.final_nb_model.fit(X_train, y_train)
        nb_preds = self.final_nb_model.predict(X_test)
        print(f"Naive Bayes Test Accuracy: {accuracy_score(y_test, nb_preds) * 100:.2f}%")
        
        self.final_rf_model.fit(X_train, y_train)
        rf_preds = self.final_rf_model.predict(X_test)
        print(f"Random Forest Test Accuracy: {accuracy_score(y_test, rf_preds) * 100:.2f}%")

    def model_fit(self, X, y):
        self.final_svm_model.fit(X, y)
        self.final_nb_model.fit(X, y)
        self.final_rf_model.fit(X, y)

    def evaluate_model(self, y_true, y_pred):
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        return classification_report(y_true, y_pred, output_dict=True)

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
    training_path = r'C:\Users\unmes\Documents\Projects\Project\disease_prediction_ml\data\Training.csv'
    test_path = r'C:\Users\unmes\Documents\Projects\Project\disease_prediction_ml\data\Testing.csv'
    
    disease = DiseasePredict()
    data = disease.read_data(training_path)
    X_train, X_test, y_train, y_test, X, y = disease.encode(data)
    
    print("Cross-validation scores:")
    disease.cross_val(X, y)
    
    print("\nTraining individual classifiers and evaluating on test set:")
    disease.train_classifiers(X_train, X_test, y_train, y_test)
    
    print("\nTraining final models on full dataset:")
    disease.model_fit(X, y)
    
    print("\nPrediction results for input symptoms:")
    predict = disease.predictDisease(symptoms="Itching,Skin Rash,Nodal Skin Eruptions")
    print(predict)
