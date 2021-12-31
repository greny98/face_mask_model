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
    image_tensor = image_tensor / 127.5 - 1.
    cls, boxes = model(image_tensor)
    return tf.image.combined_non_max_suppression(
        tf.expand_dims(boxes, axis=2),
        cls,
        max_output_size_per_class=10,
        max_total_size=10,
        iou_threshold=0.5,
        score_threshold=0.5,
        clip_boxes=False,
    )


def draw_on_frame(frame, results):
    bboxes, scores, labels, valid = [elm.numpy()[0] for elm in results]
    print(bboxes, scores, labels, valid)

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
