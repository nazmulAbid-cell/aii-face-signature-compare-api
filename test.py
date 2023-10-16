# import face_recognition
# image1 = face_recognition.load_image_file("C://Users//User//Desktop//New folder//ains.jpg")
# image2 = face_recognition.load_image_file("C://Users//User//Desktop//New folder//ains.jpg")
#
# encoding1 = face_recognition.face_encodings(image1)[0]
# encoding2 = face_recognition.face_encodings(image2)[0]
#
# result = face_recognition.compare_faces([encoding1], encoding2)
#
# if result[0]:
#     print("The two images have the same person.")
# else:
#     print("The two images have different people.")
#

import numpy as np
import face_recognition

try:
    image1 = face_recognition.load_image_file("C://Users//User//Desktop//New folder//ains.jpg")
    image2 = face_recognition.load_image_file("C://Users//User//Desktop//New folder//em.jpg")
except FileNotFoundError as e:
    print("Error loading image:", e)
    exit()

try:
    encoding1 = face_recognition.face_encodings(image1)[0]
    encoding2 = face_recognition.face_encodings(image2)[0]
except IndexError:
    print("No face detected in one or both images.")
    exit()
encodings1 = face_recognition.face_encodings(image1)
encodings2 = face_recognition.face_encodings(image2)

if not encodings1 or not encodings2:
    print("No faces detected in one or both images.")
    exit()

# Convert the lists to numpy arrays
encodings1 = np.array(encodings1)
encodings2 = np.array(encodings2)

results = face_recognition.compare_faces(encodings1, encodings2)

if any(results):
    print("matching person.")
else:
    print("different person.")



