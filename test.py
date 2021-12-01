import cv2

from configs.common_config import IMAGE_SIZE
from keras import Model
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
import time

from model.inference import PredictModel


def predict(model: Model, image):
    h, w, c = image.shape
    resized = cv2.resize(image, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
    image_tensor = image_tensor[tf.newaxis, ...]
    image_tensor = mobilenet_v2.preprocess_input(image_tensor)
    results = model(image_tensor)
    bboxes, scores, labels, valid = [elm.numpy()[0] for elm in results]
    if valid > 0:
        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            xmin = int(xmin * w / 320)
            xmax = int(xmax * w / 320)
            ymin = int(ymin * h / 320)
            ymax = int(ymax * h / 320)
            if label == 0:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            else:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    return image


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    model = tf.saved_model.load("saved_model")
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = predict(model, frame)
        cv2.imshow("frame", frame)
        print(1 / (time.time() - start))
        key = cv2.waitKey(2)
        if key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
