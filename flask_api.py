from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)

CLASS_LABELS = ['airplane',
                'bird',
                'car',
                'cat',
                'deer',
                'dog',
                'horse',
                'monkey',
                'ship',
                'truck']
IMAGE_SIZE = [96, 96]


@app.route("/classify", methods=['POST'])
def predict():
    r = request
    np_arr = np.frombuffer(r.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = load_img(img, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]))
    img = img_to_array(img)
    loaded_model = load_model("./models/inception.h5", compile=True)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    prediction = loaded_model.predict(img)
    pred = np.argmax(prediction, axis=-1)
    return jsonify(CLASS_LABELS[pred[0]])


def run_server_api():
    app.run(host='localhost', port=8080, debug=True)


if __name__ == "__main__":
    run_server_api()
