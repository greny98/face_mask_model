import tensorflow as tf
from model.inference import PredictModel

model = PredictModel(4)
model.load_weights("ckpt/face_mask/checkpoint")
tf.saved_model.save(model, "tflite_model/saved_model")