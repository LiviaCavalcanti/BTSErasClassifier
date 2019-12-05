import cv2
import os
from sklearn.model_selection import train_test_split
import shutil
import mahotas

SAMPLE_FOLDER = "../sample_images" # if you can't find images, try run this file in analysis folder
TRAIN_NAMES="trainImages.csv"
TEST_NAMES="testImages.csv"
IMG_FOLDER="../img"
TEST_FEATURES_FILE="testFeatures.csv"
TRAIN_FEATURES_FILE="trainFeatures.csv"
TEST_FOLDER="test/"
TRAIN_FOLDER="train/"
TEST_PATH = IMG_FOLDER + "/" + TEST_FOLDER
TRAIN_PATH = IMG_FOLDER + "/" + TRAIN_FOLDER

def get_histogram(image, bins=256):
    r''' returns a dict with histogram per channel of an image '''
    color = ('b','g','r')
    hist_dict = {'r': [], 'g': [], 'b': []}
    for channel, col in enumerate(color):
        hist = cv2.calcHist([image],[channel],None,[bins],[0,bins])
        hist_dict[col].extend(hist.reshape(-1))
    return hist_dict

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# get the file names from a folder
def create_train_test(folder_name):
    video_filename = []

    # create these files before run

    f = open(TRAIN_NAMES, "a")
    f_test = open(TEST_NAMES, "a")
    for (dirpath, dirnames, filename) in os.walk(folder_name):

        label = dirpath.split("/")[-1]
        if label != folder_name:
            label = [label]*len(filename)
            Xi_train, Xi_test, yi_train, yi_test = train_test_split(filename, label, test_size=0.30, random_state=42)
            
            for i in range(len(Xi_train)):
                f.write(Xi_train[i] + ";" + yi_train[i]+"\n")

            for i in range(len(Xi_test)):
                f_test.write(Xi_test[i] + ";" + yi_test[i]+"\n")
    f.close()
    f_test.close()

def split_data():

    
    test_feactures = open(TEST_FEATURES_FILE, "a")
    train_feactures = open(TRAIN_FEATURES_FILE, "a")
    os.mkdir(TEST_PATH)
    os.mkdir(TRAIN_PATH)

    with open(TRAIN_NAMES, "r") as f:
        line = f.readline()
        while(line):
            img_name, folder_name = line.split(";")
            folder_name = folder_name[:-1] # remove '\n'
            img_path = IMG_FOLDER + "/" + folder_name + "/" + img_name
            train_feactures.write(extract_feature(img_path)+";"+folder_name+"\n")
            destination = TRAIN_PATH + folder_name
            if not os.path.exists(destination):
                os.mkdir(destination)
                
            shutil.copy(img_path, destination)
            line = f.readline()
    f.close()    

    with open(TEST_NAMES, "r") as f:
        line = f.readline()
        while(line):
            img_name, folder_name = line.split(";")
            folder_name = folder_name[:-1]
            img_path = IMG_FOLDER + "/" + folder_name + "/" + img_name
            test_feactures.write(extract_feature(img_path)+";"+folder_name+"\n")
            destination = TEST_PATH + folder_name
            if not os.path.exists(destination):
                os.mkdir(destination)
                
            shutil.copy(img_path, destination)
            line = f.readline()
    f.close()    

def extract_feature(img_name):
    img = cv2.imread(img_name)
    img = cv2.resize(img,(720,480))
    hist = get_histogram(img)
    moment = fd_hu_moments(img)
    hara = fd_haralick(img)

    return str(hist)+";"+ str(list(moment))+";"+ str(list(hara))



# deleting files from previous executions
if os.path.isfile(TEST_FEATURES_FILE):
    os.remove(TEST_FEATURES_FILE)
if os.path.isfile(TRAIN_FEATURES_FILE):
    os.remove(TRAIN_FEATURES_FILE)
if os.path.exists(TEST_PATH):
    shutil.rmtree(TEST_PATH)
if os.path.exists(TRAIN_PATH):
    shutil.rmtree(TRAIN_PATH)
if os.path.isfile(TEST_NAMES):
    os.remove(TEST_NAMES)
if os.path.isfile(TRAIN_NAMES):
    os.remove(TRAIN_NAMES)
# separate IMG_FOLDER images between train and test .csv
create_train_test(IMG_FOLDER)

# extract features and create folders with test and train images
split_data()