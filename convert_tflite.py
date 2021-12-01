from os.path import exists
import tensorflow as tf
import pathlib

from model.ssd import create_ssd_model

if __name__ == '__main__':
    if not exists("ckpt/checkpoint"):
        raise FileNotFoundError
    ssd_model = create_ssd_model(4)
    ssd_model.load_weights("ckpt/checkpoint").expect_partial()
    tf.saved_model.save(ssd_model, "tflite_model/saved_model")
    # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir="tflite_model/saved_model")
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_quant_model = converter.convert()
    # tflite_models_dir = pathlib.Path("tflite_model")
    # tflite_models_dir.mkdir(exist_ok=True, parents=True)
    # tflite_model_file = tflite_models_dir / "model.tflite"
    # tflite_model_file.write_bytes(tflite_quant_model)