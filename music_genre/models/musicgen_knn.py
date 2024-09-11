import numpy as np # linear algebra
import pandas as pd 
import csv
import os
from sklearn.model_selection import train_test_split



# Read input file

features = []
names = []
with open(r'C:\Users\unmes\Downloads\musicgen\Data\features_30_sec.csv','rt')as f:
    data = csv.reader(f)
    for row in data:
        features.append(np.array(row[1:]))
features = np.array(features)




key = features[0]
features = features[1:,:]
labels = features[:,-1]
categories = list(set(labels))

features = features[:,:-1]
key = key[:-1]

print('Features -', features.shape, key)
print('Categories -', categories, len(categories))

labels_temp = []
for i in range(len(categories)):
    for label in labels:
        if label == categories[i]:
            labels_temp.append(i)
labels = np.array(labels_temp)





features = features.astype(np.float)
print(features[0])

features_copy = features.copy()
for i in range(len(key)):
    Xi = features[:,i]
    features_copy[:,i] = (Xi-np.mean(Xi))/np.std(Xi)
    
from sklearn.utils import shuffle

X, Y = shuffle(features_copy, labels)

train_split = 1
X_Train, X_Val, Y_Train, Y_Val = train_test_split(X, Y, test_size=0.3)

print("X shape: {} , Y shape: {}, X train shape: {} , Y train shape: {} , X test shape: {} , Y Test Shape: {}"
      .format(X.shape, Y.shape, X_Train.shape, Y_Train.shape, X_Val.shape, Y_Val.shape))


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
X_Train= st_x.fit_transform(X_Train)    
X_Test= st_x.transform(X_Val) 




#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
#matplotlib inline


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1   
#got lowest at k=5
#K=10
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_Train, Y_Train)  #fit the model
    pred=model.predict(X_Test) #make prediction on test set
    error = sqrt(mean_squared_error(Y_Val,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


#print(pred.shape)



#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
knn.fit(X_Train, Y_Train)

prediction = knn.predict(X_Val)
prediction
from sklearn.metrics import classification_report
print(classification_report(Y_Val, prediction))


from sklearn.metrics import confusion_matrix
a=confusion_matrix(Y_Val, prediction) 
print(a)

from sklearn.metrics import accuracy_score
acc_score=accuracy_score(prediction,Y_Val)
print("the accuracy score for knn is:{}".format(acc_score))
#print("Target: {}, Predicted label: {}, Target_value:{}, Output_value:{} ".format(Y, pred, Y, pred))


                  
