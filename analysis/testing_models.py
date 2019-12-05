import pandas
import cv2
from sklearn.svm import SVC
import numpy as np
import ast
from sklearn.metrics import confusion_matrix

TRAIN_FEATURES_FILE="trainFeatures.csv"
TEST_FEATURES_FILE="testFeatures.csv"
LABELS = {"2_cool_4_school":0,"dark_n_wild":1,"hwayangyeonhwa-pt1":10,"hwayangyeonhwa-pt2":2,"love yourself tear":3,"map of the soul persona":4,"o_rul8_2":5,"skool_luv_affair":6,"wings":7,"you never walk alone":8,"young_forever":9, "love yourself her":11, "love yourself answer":12}

def read_data(file_name):
    features = []
    labels = []

    with open(file_name, "r") as f:
        line = f.readline()
        while(line):
            file_line = line.split(";")
            l, file_line = file_line[-1], file_line[:-1]
            true_features = []

            for i in file_line:
                temp = ast.literal_eval(i)
                if isinstance(temp, dict):

                    true_features.extend(list(temp.values())[0])
             
            features.append(true_features)
            labels.append(LABELS[l[:-1]])
            
            line = f.readline()
    f.close()
    return np.array(features), np.array(labels)



X_train, y_train = read_data(TRAIN_FEATURES_FILE)
X_test, y_test = read_data(TEST_FEATURES_FILE)


clf = SVC(random_state=7,gamma='scale', decision_function_shape='ovo')
clf.fit(X_train, y_train)

y_ = clf.predict(X_test)

print(confusion_matrix(y_test, y_, list(LABELS.values())))
print("accuracy: ", (sum(y_test==y_))/len(y_))