import cv2 as cv


class FaceDetector:
    def __init__(self, path):
        self.face_cascade = cv.CascadeClassifier(path)

    def detect(self, image, scale_factor=1.1, min_neighbors=5, min_size=(100, 100)):
        faces = self.face_cascade.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                                   minSize=min_size)
        return faces
