import tensorflow as tf
from keras.models import Model

from configs.common_config import IMAGE_SIZE, IMAGE_SIZE_LARGE
from model.anchor_boxes import PredictionDecoder
from model.ssd import create_ssd_model


def PredictModel(num_classes, large, weights=None):
    ssd_model = create_ssd_model(num_classes, large)
    if weights is not None:
        ssd_model.load_weights(weights)
    if large:
        image = tf.keras.Input(shape=(IMAGE_SIZE_LARGE, IMAGE_SIZE_LARGE, 3), name="image")
    else:
        image = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    predictions = ssd_model(image, training=False)
    detections = PredictionDecoder()(image, predictions)
    inference_model = Model(inputs=image, outputs=detections)
    return inference_model
