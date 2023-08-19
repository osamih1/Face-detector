import os
import cv2
import matplotlib.pyplot as plt

data_dir = './data'
modes_dir = './models'

face_detector = cv2.CascadeClassifier(os.path.join(modes_dir, 'haarcascade_frontalface_default.xml'))

for img_path in os.listdir(data_dir):
    print(img_path)
    img = cv2.imread(os.path.join(data_dir, img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, minNeighbors=20)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    for face in faces:
        x1, y1, w, h = face
        cv2.rectangle(img, (x1,y1), (x1+w, y1+h), (0,255,0), 10)
    plt.imshow(img)
plt.show()