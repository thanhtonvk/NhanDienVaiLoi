import cv2
from modules.object_detection import ObjectDetecion
from modules.classifier import Classifier

detector = ObjectDetecion()
classifier = Classifier()


def predict(image):
    # img_width, img_height, _ = image.shape
    image_org = image
    boxes,scores = detector.detect(image)
    crop = None
    for idx,box in enumerate(boxes):
        box = list(map(int, box))
        x_min, y_min, x_max, y_max = box
        cv2.putText(image, str(scores[idx]), (x_max, y_max), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, ), 3)
        crop = image_org[y_min:y_max, x_min:x_max]
        try:
            pred = classifier.predict(crop)
            cv2.putText(image, str(pred[0]) + '-' + str(pred[1]), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
        except Exception as e:
            print(e)
    return image


if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    while (True):
        ret, image = vid.read()
        if ret is None:
            break
        image = predict(image)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    
