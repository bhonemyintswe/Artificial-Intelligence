from face_recognizer import FaceRecognizer
import numpy as np
import glob
import cv2 as cv

faces_path = "./output/faces/*.npy"
output_path = "./output/classifier"
sample_size = 100

fr = FaceRecognizer(cv.face_LBPHFaceRecognizer.create(radius=1, neighbors=8, grid_x=8, grid_y=8))
labels = []
faces = []

for (i, path) in enumerate(glob.glob(faces_path)):
    name = path[path.rfind("/") + 1: -4]
    # print(f"I is {i} and path is {path} name is {name}")

    sample = np.load(path)
    # print(f'sample is {sample}')
    # sample = sample[np.random.choice(sample.shape[0], 10, replace=False), : ]
    # sample = open(path).read().strip().split("\n")
    # sample = random.sample(sample, min(len(sample), sample_size))

    # for face in sample:

    # print(f"faces is {faces}")
    faces.append(sample)

    labels.append(name)
    train_labels = np.array([i] * len(faces[i]))
    fr.train(faces[i], train_labels)

fr.set_labels(labels)
fr.save(output_path)
