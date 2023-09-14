import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np


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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='models/inception_resnet_v2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/dataset/unlabeled/0.jpg', help='file/dir/URL/glob')
    opt = parser.parse_args()
    return opt


def predict(model, source):
    img = load_img(source, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]))
    img = img_to_array(img)
    loaded_model = load_model(model, compile=True)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    prediction = loaded_model.predict(img)
    pred = np.argmax(prediction, axis=-1)
    print(CLASS_LABELS[pred[0]])


def main(opt):
    predict(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
