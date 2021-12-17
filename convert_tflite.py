import pathlib

import pandas as pd
import tensorflow as tf

from configs.common_config import IMAGE_SIZE
from data_utils.mask_generator import create_image_info
from model.inference import PredictModel
from model.ssd import create_ssd_model


def preprocess(image_file):
    image_raw = tf.io.read_file(image_file)
    img = tf.image.decode_jpeg(image_raw, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img / 127.5 - 1.
    return img


def representative_data_gen():
    ds = tf.data.Dataset.from_tensor_slices(image_files).map(preprocess).batch(1).take(100)
    for input_value in ds:
        yield [input_value]


if __name__ == '__main__':
    info = create_image_info('data/kaggle_mask', 'data/medical_mask')
    train_images = pd.read_csv('train.csv')['filename'].values
    train_info = {
        filename: info
        for filename, info in info.items() if filename.split('/')[-1] in train_images}
    image_files = [filename for filename in train_info.keys()]
    # if not exists("ckpt/checkpoint"):
    #     raise FileNotFoundError
    model = PredictModel(4)
    # model = create_ssd_model(4)

    tf.saved_model.save(model, "tflite_model/saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir='tflite_model/saved_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()
    tflite_models_dir = pathlib.Path("tflite_model")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir / "model.tflite"
    tflite_model_file.write_bytes(tflite_quant_model)
