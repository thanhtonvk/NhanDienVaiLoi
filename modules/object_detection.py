import tensorflow as tf
import numpy as np
import cv2


def preprocessing(image, input_size=640):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size),
                       interpolation=cv2.INTER_AREA).astype('float32')
    image /= 255.0
    return image


def load_model(model_path):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


class ObjectDetecion:
    def __init__(self, input_size=640, model_path='models/best-fp16.tflite'):
        self.input_size = input_size
        self.model_path = model_path
        self.interpreter, self.input_details, self.output_details = load_model(
            model_path=self.model_path)

    def detect(self, image, threshold=0.3):
        img_width, img_height, _ = image.shape
        input_data = np.expand_dims(
            preprocessing(image, self.input_size), axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(
            self.output_details[-1]['index'])[0]

        boxes = [
            boxes[i]
            for i, score in enumerate(scores)
            if score > threshold
        ]
        scores = [
            scores[i]
            for i, score in enumerate(scores)
            if score > threshold
        ]

        boxes = np.array(boxes)
        if len(boxes) > 0:
            # resize bbox to raw size
            boxes[:, 0:1] *= img_height
            boxes[:, 1:2] *= img_width
            boxes[:, 2:3] *= img_height
            boxes[:, 3:4] *= img_width
        return boxes,scores


if __name__ == '__main__':
    detector = ObjectDetecion()
    vid = cv2.VideoCapture(0)

    while (True):

        # Capture the video frame
        # by frame
        ret, image = vid.read()
        boxes = detector.detect(image)
        for box in boxes:
            box = list(map(int, box))
            x_min, y_min, x_max, y_max = box
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
        cv2.imshow('image', image)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
