'''This face recognition system is used to recognize only one person'''
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
'''Important note : All the data of images all handled only in greyscale that's why we use many time convert the raw image into grey image'''
data_path='D:/My Files/Face Recognition/faces/'
'''the data_path  contains the location of the images that we captured for face recognition module '''
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
#this is way of list comprehension
Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
#the for loop is for collecting the images from the folder and make it available in the list
Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
#LBPHP - Local Binary Pattern Histogram Face Recognizer model
model.train(np.asarray(Training_Data), np.asarray(Labels))
#the train method is the most important part of the code ,its only is responsible for training the data
print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]
    '''the if condition is to check whether the face is available in the screen or not'''
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+'% Piyush'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Recognizer', image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognizer', image)


    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognizer', image)
        pass
    #press l to stop the video streaming 
    if cv2.waitKey(1) & 0xFF == ord('l'):
        break


cap.release()
cv2.destroyAllWindows()