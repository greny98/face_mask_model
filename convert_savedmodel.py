import tensorflow as tf
from model.inference import PredictModel

model = PredictModel(1, large=True, weights="ckpt/fire/checkpoint")
tf.saved_model.save(model, "saved_model")