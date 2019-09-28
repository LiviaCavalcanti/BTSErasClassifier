import cv2
import os

MV_FOLDER = "mv_imgs/" # where images from mv are saved in
INIT_VIDEO = 190 # how many frames are skipped to finally save one
BTS_VIDEO_FOLDER = './bts' # where mvs are saved

# get the file names from a folder
def read_video_names(video_folder):
    video_filename = []
    for (dirpath, dirnames, filename) in os.walk(video_folder):
        video_filename.extend(filename)

    return(video_filename)


img_folder = read_video_names(BTS_VIDEO_FOLDER)

for filename in img_folder:

    # reading video from its name in folder
    video_name = BTS_VIDEO_FOLDER + '/' + filename 
    mv_name = filename[:-4]

    vs = cv2.VideoCapture(video_name)
    firstFrame = None
    
    # countroling frames number
    count = 0
    i = 0

    # creating folder for each mv
    if not os.path.exists(MV_FOLDER + mv_name):
        os.makedirs(MV_FOLDER + mv_name)

    # playing video
    while True:
        frame = vs.read()
        frame = frame[1]    
        if frame is None:
            break

        if count % 80 == 0 and count > INIT_VIDEO: # saving pictures
            cv2.imwrite(MV_FOLDER + mv_name + "/" + mv_name + "_" + str(i) + ".jpg", frame)
            i+=1
        count +=1
