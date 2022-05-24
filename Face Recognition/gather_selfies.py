import imutils
import cv2 as cv
import numpy as np
from face_detector import FaceDetector

name = input("Enter your name : ")
method = input("Method 1 = gather_selfies with camera, \n"
               "Method 2 = using the video, \n"
               "Input (1 or 2):")

path = '../../haarcascades/haarcascade_frontalface_alt2.xml'
f_detector = FaceDetector(path)
captureMode = False
skip = 0
color = (0, 255, 0)
face_data = []
f = open(f'output/faces/{name}.npy', 'w')
face = np.zeros((100, 100), dtype="uint8")
total = 0

if method == "1":
    cap = cv.VideoCapture(0)
else:
    cap = cv.VideoCapture(f"input/{name}.mp4")
    captureMode = True
    color = (0, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = f_detector.detect(gray, scale_factor=1.1, min_neighbors=5, min_size=(100, 100))

    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])

        if captureMode:
            face = gray[y:y + h, x:x + w].copy(order="C")
            face_resize = cv.resize(face, (100, 100))
            # face_data.append(face_resize)

            if skip % 10 == 0:
                face_data.append(face_resize)
            skip += 1
            total = skip

        cv.rectangle(frame, (x, y), (x + w, y + h), color)

    cv.imshow("Camera", frame)
    key_press = cv.waitKey(1)
    if key_press == ord('q'):
        break

    elif key_press == ord('c'):
        if captureMode:
            captureMode = False
            color = (0, 255, 0)
        else:
            captureMode = True
            color = (0, 0, 255)

# nested list to 3 dimensional numpy array
face_data = np.asarray(face_data)
# # 3 dimensional numpy array to 2 dimensional numpy array
# face_data = face_data.reshape((face_data.shape[0], -1))
np.save('./output/faces/' + name + '.npy', face_data)

print(f'[INFO] wrote {int(total)} frames to file')
f.close()
cap.release()
cv.destroyAllWindows()
