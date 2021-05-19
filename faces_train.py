import os
import cv2 as cv
import numpy as np

people = ['Amitabh Bachan','Akshay Kumar','Shradha Kapoor','Katrina Kaif','Sonu Sood']

DIR = r'C:\Users\hp\Desktop\photos-cv'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

feature = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)


        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                feature.append(faces_roi)
                labels.append(label)

create_train()

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer

feature = np.array(feature, dtype='object')
labels = np.array(labels)

face_recognizer.train(feature,labels)
face_recognizer.save('face_trained.yml')

np.save('feature.npy',feature)
np.save('labels.npy',labels)


