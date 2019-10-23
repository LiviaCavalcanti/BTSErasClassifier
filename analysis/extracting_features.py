import cv2
import os

SAMPLE_FOLDER = "../sample_images" # if you can't find images, try run this file in analysis folder

def get_histogram(image, bins=256):
    r''' returns a dict with histogram per channel of an image '''
    color = ('b','g','r')
    hist_dict = {'r': [], 'g': [], 'b': []}
    for channel, col in enumerate(color):
        hist = cv2.calcHist([image],[channel],None,[bins],[0,bins])
        hist_dict[col].extend(hist.reshape(-1))
    return hist_dict

# get the file names from a folder
def read_video_names(video_folder):
    video_filename = []
    for (dirpath, dirnames, filename) in os.walk(video_folder):

        video_filename.extend(filename)

    return(video_filename)


img_folder = read_video_names(SAMPLE_FOLDER)

for img_name in img_folder:
    img = cv2.imread(SAMPLE_FOLDER + "/" +img_name)
    hist = get_histogram(img)
    cv2.imshow('i', img)
    cv2.waitKey(0)
    break
