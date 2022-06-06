import os
import cv2 as cv
import sysconfig


path = sysconfig.get_paths()['purelib'] + '/cv2/data/'
faceCascade = cv.CascadeClassifier(path
                    + 'haarcascade_frontalface_default.xml')
os.chdir('foto')
contents = sorted(os.listdir())

for foto in contents:
    foto = cv.imread(foto)
    gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=6,
        minSize=(30, 30))
    print("Znaleziono {0} twarzy!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv.rectangle(foto, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow("Wykrywacz twarzy", foto)
    cv.waitKey(2000)
