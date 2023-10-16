import numpy as np
import face_recognition
from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)

similarity_threshold = 0.8


def compare_faces(image1_path, image2_path):
    try:
        image1 = face_recognition.load_image_file(image1_path)
        image2 = face_recognition.load_image_file(image2_path)
    except FileNotFoundError as e:
        return "Error loading image: " + str(e), None

    encodings1 = face_recognition.face_encodings(image1)
    encodings2 = face_recognition.face_encodings(image2)

    if not encodings1 or not encodings2:
        return "No faces detected in one or both images.", None

    encodings1 = np.array(encodings1)
    encodings2 = np.array(encodings2)

    results = face_recognition.compare_faces(encodings1, encodings2)

    if any(results):
        return "matching person", True
    else:
        return "different person", False


def compare_signatures(signature1, signature2):
    image1 = cv2.imdecode(np.fromstring(signature1.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imdecode(np.fromstring(signature2.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matcher = cv2.FlannBasedMatcher()
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    similarity = len(good_matches) / max(len(keypoints1), len(keypoints2))
    return similarity


@app.route('/compare_faces', methods=['POST'])
def api_compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Both 'image1' and 'image2' must be provided in the request."}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    result, match = compare_faces(image1, image2)
    return jsonify({"result": result, "match": match})


@app.route('/compare_signatures', methods=['POST'])
def compare_signatures_endpoint():
    try:
        if 'signature1' not in request.files or 'signature2' not in request.files:
            return jsonify({"error": "Both 'signature1' and 'signature2' files are required."}), 400

        similarity = compare_signatures(request.files['signature1'], request.files['signature2'])

        if similarity > similarity_threshold:
            response = {"result": "same_signature"}
        else:
            response = {"result": "different_signatures"}

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
