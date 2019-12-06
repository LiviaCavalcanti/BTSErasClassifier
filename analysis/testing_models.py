import pandas
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import ast
from sklearn.metrics import confusion_matrix, accuracy_score
from extracting_features import data_eng
import pandas as pd

TRAIN_FEATURES_FILE="trainFeatures.csv"
TEST_FEATURES_FILE="testFeatures.csv"
LABELS = {'young_forever':0, '2_cool_4_school':1, 'skool_luv_affair':2, 'dark_n_wild':3, 'o_rul8_2':4, 'you never walk alone':5}

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
                    for i in list(temp.values()):
                        true_features.extend(i)
             
            features.append(true_features)
            labels.append(LABELS[l[:-1]])
            
            line = f.readline()
    f.close()
    return np.array(features), np.array(labels)

def test_model(model_name = "svm"):
    # preparing data
    data_eng()

    X_train, y_train = read_data(TRAIN_FEATURES_FILE)
    X_test, y_test = read_data(TEST_FEATURES_FILE)
    X_val, y_val = read_data("validation.csv")

    # loading model
    if model_name == "svm":
        clf = SVC(random_state=7,gamma='scale', decision_function_shape='ovo')
    elif model_name == "knn":
        clf = KNeighborsClassifier(n_neighbors=15)
    
    # training model
    clf.fit(X_train, y_train)
    y_ = clf.predict(X_test)
    pred_val = clf.predict(X_val)

    # evaluating model
    cm_test = confusion_matrix(y_test, y_, list(LABELS.values()))
    cm_val = confusion_matrix(y_val, pred_val, list(LABELS.values()))
    print(">>> Evaluating results for " + model_name + " model")
    print(">>> Train Accuracy: ", accuracy_score(y_train, clf.predict(X_train)))
    print(">>> Test Accuracy: ", accuracy_score(y_test, y_))
    print(">>> Validation Accuracy: ", accuracy_score(y_val, pred_val))
    print("\n====== CONFUSION MATRIX ======")
    
    dataset_test = pd.DataFrame({'Young Forever': cm_test[:, 0], '2 Cool 4 School': cm_test[:, 1], 'Skool Luv Affair': cm_test[:, 2], 'Dark N Wild': cm_test[:, 3], 'o r u l8 2': cm_test[:, 4], 'You Never Walk Alone': cm_test[:, 5]})
    dataset_val = pd.DataFrame({'Young Forever': cm_val[:, 0], '2 Cool 4 School': cm_val[:, 1], 'Skool Luv Affair': cm_val[:, 2], 'Dark N Wild': cm_val[:, 3], 'o r u l8 2': cm_val[:, 4], 'You Never Walk Alone': cm_val[:, 5]})
    print("\nTest Confusion Matrix")
    print(dataset_test)
    print("\nValidation Confusion Matrix")
    print(dataset_val)

test_model()