from os.path import join
import numpy as np
import tensorflow as tf

from data_utils.data_generator import detect_augmentation
from data_utils.kaggle_mask import read_kaggle_mask
from data_utils.medical_mask import read_medical_mask

AUTOTUNE = tf.data.AUTOTUNE


def create_image_info(kaggle_dir, medical_dir):
    info = read_kaggle_mask(join(kaggle_dir, 'annotations'), join(kaggle_dir, 'images'), {})
    info = read_medical_mask(join(medical_dir, 'annotations'), join(medical_dir, 'images'), info)
    return info


def MaskGenerator(images_info: dict, label_encoder, object_names, training=True, batch_size=10):
    # Extract infomation from images_info
    image_files = [filename for filename in images_info.keys()]
    bboxes = [list(image['bboxes']) for image in images_info.values()]
    labels = [list(image['labels']) for image in images_info.values()]
    # padding boxes
    pad_bbox = np.zeros(4, dtype=np.float32)
    pad_label = -1
    num_bboxes = [len(label) for label in labels]
    max_padding = max(num_bboxes)
    for i in range(len(bboxes)):
        for _ in range(num_bboxes[i], max_padding):
            bboxes[i].append(pad_bbox)
            labels[i].append(pad_label)
    # Create tensor slices
    image_files_slices = tf.data.Dataset.from_tensor_slices(image_files)
    bboxes_slices = tf.data.Dataset.from_tensor_slices(bboxes)
    labels_slices = tf.data.Dataset.from_tensor_slices(labels)
    num_bboxes_slices = tf.data.Dataset.from_tensor_slices(num_bboxes)
    y_slices = tf.data.Dataset.zip((bboxes_slices, labels_slices, num_bboxes_slices))

    # Create dataset with process
    def process_data(image_file, y, training):
        aug_image, labels = tf.numpy_function(func=detect_augmentation(label_encoder, training, object_names),
                                              inp=[image_file, y[0], y[1], y[2]],
                                              Tout=[tf.float32, tf.float32])
        return aug_image, labels

    ds = tf.data.Dataset.zip((image_files_slices, y_slices))

    ds = ds.shuffle(512, reshuffle_each_iteration=training)
    ds = ds.map(lambda x, y: process_data(x, y, training=training), num_parallel_calls=AUTOTUNE).batch(
        batch_size).prefetch(AUTOTUNE)
    return ds
