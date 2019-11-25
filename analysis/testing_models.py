import pandas
import cv2
from sklearn import svm, preprocessing
import numpy as np
import ast

TRAIN_FEATURES_FILE="trainFeatures.csv"

clf =svm.SVC(random_state=9)
features = []

with open(TRAIN_FEATURES_FILE, "r") as f:
    line = f.readline()
    while(line):
        # features.append(ast.literal_eval(line.split(";")[:-1]))
        file_line = line.split(";")[:-1]
        true_features = []
        for i in file_line:
            temp = ast.literal_eval(i)
            
            if isinstance(temp, dict):
                print("+++++++++++++++++++++++++")
                true_features.extend(list(temp.values()))
            else:
                true_features.append(temp)
        
        features.append(true_features)
        line = f.readline()
f.close()

print("------------------------------")
# global_feature = np.hstack(features)
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# #Normalize The feature vectors...
# rescaled_features = scaler.fit_transform(global_feature)
# print(":::::::::::::::::::::::::::::::::::::")
prediction= clf.fit(features)[0]
print(prediction)
