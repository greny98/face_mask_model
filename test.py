import cv2

from configs.common_config import IMAGE_SIZE
from keras import Model
import tensorflow as tf
from tensorflow.keras.applications import densenet
import time

from model.inference import PredictModel


@tf.function
def predict(model: Model, image_tensor):
    image_tensor = tf.image.resize(image_tensor, size=(IMAGE_SIZE, IMAGE_SIZE))
    image_tensor = image_tensor[tf.newaxis, ...]
    # image_tensor /= 255.
    image_tensor = densenet.preprocess_input(image_tensor)
    results = model(image_tensor)
    return results


def draw_on_frame(frame, results):
    bboxes, scores, labels, valid = [elm.numpy()[0] for elm in results]
    h, w, _ = frame.shape
    if valid > 0:
        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            xmin = int(xmin * w / IMAGE_SIZE)
            xmax = int(xmax * w / IMAGE_SIZE)
            ymin = int(ymin * h / IMAGE_SIZE)
            ymax = int(ymax * h / IMAGE_SIZE)
            if label == 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            else:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    model = tf.saved_model.load("saved_model")
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        image_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        results = predict(model, image_tensor)
        frame = draw_on_frame(frame, results)
        cv2.imshow("frame", frame)
        print(1 / (time.time() - start))
        key = cv2.waitKey(2)
        if key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
