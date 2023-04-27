import cv2
import numpy as np
import tensorflow as tf

labels = {
    0: 'Loang mau',
    1: 'Rach',
    2: 'Rut soi',
    3: 'Xuoc'
}


def preprocessing(image, input_size=224):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size)).astype('float32')
    image /= 127.5 - 1
    return image


class Classifier:
    def __init__(self, input_size=224, path='models/bestmodel.h5'):
        self.input_size = input_size
        self.path = path
        self.model = tf.keras.models.load_model(self.path)

    def predict(self, image):
        image = preprocessing(image, input_size=self.input_size)
        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image, 0)
        print(predictions)
        idx = np.argmax(predictions)
        return labels[idx], int(predictions[0][idx] * 100)


if __name__ == '__main__':
    classify = Classifier()
    image = cv2.imread('102.jpg')
    print(classify.predict(image))
