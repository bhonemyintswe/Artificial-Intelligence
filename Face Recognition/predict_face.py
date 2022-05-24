import cv2 as cv
import imutils
from face_recognizer import FaceRecognizer
from face_detector import FaceDetector

cascade_path = '../../haarcascades/haarcascade_frontalface_alt2.xml'
classifier_path = './output/classifier'
confidence = 100
f_detector = FaceDetector(cascade_path)
f_recognizer = FaceRecognizer.load(classifier_path)
f_recognizer.set_confidence_threshold(confidence)

method = input("Method 1 = gather_selfies with camera, \n"
               "Method 2 = using the video, \n"
               "Input (1 or 2):")
if method == "1":
    cap = cv.VideoCapture(0)
else:
    name = input("Enter video name: ")
    cap = cv.VideoCapture(f"input/{name}.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = f_detector.detect(gray, scale_factor=1.1, min_neighbors=5, min_size=(100, 100))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        (prediction, confidence) = f_recognizer.predict(face)
        prediction = f"{prediction}: {round(confidence, 2)}"
        cv.putText(frame, prediction, (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Camera", frame)
    k = cv.waitKey(1)

    if k == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

