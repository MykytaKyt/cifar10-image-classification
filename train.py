import argparse
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.plots import plot_loss, plot_acc
from utils.save_model import save
from utils.dataset import get_path

IMAGE_SIZE = [96, 96]


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='data/10_class.yaml', help='config yaml path')
    parser.add_argument('--model', type=str,
                        default='inception', help='inception or xception or efficientnet')
    parser.add_argument('--name', type=str,
                        default='model_run_1', help='model name')
    opt = parser.parse_args()
    return opt


def processing_data(data):
    train_path, test_path = get_path(data)
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.3,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(96, 96),
                                                     batch_size=16,
                                                     class_mode='sparse')
    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size=(96, 96),
                                                batch_size=16,
                                                class_mode='sparse')

    return training_set, test_set


def run(data, model, name):
    training_set, test_set = processing_data(data)
    global transfer
    if model == 'inception':
        transfer = InceptionResNetV2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    elif model == 'xception':
        transfer = Xception(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    elif model == 'efficientnet':
        transfer = EfficientNetB7(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    else:
        print("Choose correct model")

    for layer in transfer.layers:
        layer.trainable = False
    x = Flatten()(transfer.output)
    prediction = Dense(10, activation='softmax')(x)
    model = Model(inputs=transfer.input, outputs=prediction)
    model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )
    r = model.fit(
      training_set,
      validation_data=test_set,
      epochs=15,
      steps_per_epoch=len(training_set),
      validation_steps=len(test_set))
    plot_loss(r)
    plot_acc(r, name)
    train_scores = model.evaluate(training_set, batch_size=16)
    test_scores = model.evaluate(test_set, batch_size=16)
    print(f"Final accuracy on train set: {train_scores[1]*100:.2f}%")
    print(f"Final accuracy on test set: {test_scores[1]*100:.2f}%")
    save(model, name)


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
