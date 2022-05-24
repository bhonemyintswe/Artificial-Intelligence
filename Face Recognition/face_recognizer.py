from collections import namedtuple
import pickle
import cv2 as cv
import os

FaceRecognizerInstance = namedtuple("FaceRecognizerInstance", ["trained", "labels"])


class FaceRecognizer:
    def __init__(self, recognizer, trained=False, labels=False):
        self.recognizer = recognizer
        self.trained = trained
        self.labels = labels

    def set_labels(self, labels):
        self.labels = labels

    def train(self, data, labels):
        if not self.trained:
            self.recognizer.train(data, labels)
            self.trained = True
            return
        # print(f"labels are {labels}")

        self.recognizer.update(data, labels)

    def save(self, base_path):
        # construct the face recognizer instance
        fri = FaceRecognizerInstance(trained=self.trained, labels=self.labels)

        # due to strange behavior with OpenCV, we need to make sure the output classifier file
        # exists prior to writing it to file
        if not os.path.exists(base_path + "/classifier.model"):
            f = open(base_path + "/classifier.model", "w")
            f.close()

        # write the actual recognizer along with the parameters to file
        self.recognizer.save(base_path + "/classifier.model")
        f = open(base_path + "/pickle", "wb")
        f.write(pickle.dumps(fri))
        f.close()

    @staticmethod
    def load(base_path):
        fri = pickle.loads(open(base_path + "/pickle", "rb").read())
        recognizer = cv.face_LBPHFaceRecognizer.create()
        recognizer.read(base_path + "/classifier.model")

        return FaceRecognizer(recognizer, trained=fri.trained, labels=fri.labels)

    def set_confidence_threshold(self, confidence_threshold):
        self.recognizer.setThreshold(confidence_threshold)

    def predict(self, face):
        (prediction, confidence) = self.recognizer.predict(face)
        if prediction == -1:
            return "Unknown", 0

        return self.labels[prediction], confidence
