import cv2
import os


SAMPLE_FOLDER = "../sample_images" # if you can't find images, try run this file in analysis folder


# get the file names from a folder
def read_video_names(video_folder):
    video_filename = []
    for (dirpath, dirnames, filename) in os.walk(video_folder):

        video_filename.extend(filename)

    return(video_filename)


img_folder = read_video_names(SAMPLE_FOLDER)

for img_name in img_folder:
    img=cv2.imread(SAMPLE_FOLDER + "/" +img_name)
    cv2.imshow('i', img)
    cv2.waitKey(0)
