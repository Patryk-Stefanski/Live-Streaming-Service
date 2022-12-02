import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
images = os.path.join(BASE_DIR, "sample-images")

face_cascade = cv2.CascadeClassifier('faceDetectionModels/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

uid = 0
label_ids = {}
labels = []
faceAOIs = []

# Loops through all the directories and files in the sample-images directory
for root, dirs, files in os.walk(images):
    # Loops through all the files in each folder of the sample-images directory
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)

            # If file hasn't already been compressed then change its size, compress it,
            # save it with a new distinguishable name and remove the old file
            if not file.startswith('compressed'):
                nonCompressedImage = Image.open(path)
                size = (250, 250)
                image = nonCompressedImage.resize(size, Image.ANTIALIAS)
                image.save(root + "/compressed_" + file, optimize=True, quality=30)
                os.remove(path)
                path = os.path.join(root, "compressed_" + file)

            # Check if the label is in our set of labels , if not then add it and increase uid
            label = os.path.basename(root).replace(" ", "-").lower()  # use sample-images sub dir as names for labels
            if not label in label_ids:
                label_ids[label] = uid
                uid += 1
            id_ = label_ids[label]  # match the id with the already existing label
            pil_image = Image.open(path).convert("L")  # grayscale
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=6)  # detect faces in pic

            for (x, y, w, h) in faces:
                aoi = image_array[y:y + h, x:x + w]  # get area of interest from picture
                faceAOIs.append(aoi)  # add aoi to array
                labels.append(id_)  # add the id, so it can relate a label to an aoi of a pic

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(faceAOIs, np.array(labels))
recognizer.save("recognizers/face-trainer.yml")
