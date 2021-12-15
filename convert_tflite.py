import pathlib

import tensorflow as tf

from model.inference import PredictModel

if __name__ == '__main__':
    # if not exists("ckpt/checkpoint"):
    #     raise FileNotFoundError
    model = PredictModel(4)

    tf.saved_model.save(model, "tflite_model/saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir="tflite_model/saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()
    tflite_models_dir = pathlib.Path("tflite_model")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir / "model.tflite"
    tflite_model_file.write_bytes(tflite_quant_model)
