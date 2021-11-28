from os.path import join

import tensorflow as tf
import numpy as np
import albumentations as augment
from tensorflow.keras.applications import densenet

from datasets.kaggle_mask import read_kaggle_mask
from datasets.medical_mask import read_medical_mask
from datasets.values import object_names
from model.anchor_boxes import LabelEncoder

IMAGE_SIZE = 512
AUTOTUNE = tf.data.AUTOTUNE


def create_image_info(kaggle_dir, medical_dir):
    info = read_kaggle_mask(join(kaggle_dir, 'annotations'), join(kaggle_dir, 'images'), {})
    info = read_medical_mask(join(medical_dir, 'annotations'), join(medical_dir, 'images'), info)
    return info


def detect_augmentation(label_encoder: LabelEncoder, training: bool):
    if training:
        transform = augment.Compose([
            augment.ImageCompression(quality_lower=80, quality_upper=100),
            augment.HorizontalFlip(),
            augment.VerticalFlip(),
            # augment.RandomRotate90(),
            augment.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            augment.ShiftScaleRotate(shift_limit=0.015, scale_limit=0.015, rotate_limit=25),
            augment.GaussNoise(),
            augment.RandomSizedBBoxSafeCrop(IMAGE_SIZE, IMAGE_SIZE),
        ], bbox_params=augment.BboxParams(format='pascal_voc'))
    else:
        transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)],
                                    bbox_params=augment.BboxParams(format='pascal_voc'))

    def preprocess_image(image_file, bboxes, labels, n_bbox):
        image_raw = tf.io.read_file(image_file)
        trans_bboxes = []
        bboxes = bboxes[:n_bbox]
        labels = labels[:n_bbox]
        decoded = tf.image.decode_jpeg(image_raw, channels=3)
        h, w, c = decoded.numpy().shape
        for i, bbox in enumerate(bboxes[:n_bbox]):
            bbox[0] = np.minimum(bbox[0], w)
            bbox[2] = np.minimum(bbox[2], w)
            bbox[1] = np.minimum(bbox[1], h)
            bbox[3] = np.minimum(bbox[3], h)
            if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                continue
            trans_bbox = list(bbox)
            trans_bbox.append(object_names[labels[i] - 1])
            trans_bboxes.append(trans_bbox)
        if len(trans_bboxes) == 0:
            print("==============")
            print(image_file)
            print("==============")
        data = {'image': decoded.numpy(), 'bboxes': trans_bboxes}
        transformed = transform(**data)
        # extract transformed image
        aug_img = transformed['image']
        aug_img = tf.cast(aug_img, tf.float32)
        aug_img = densenet.preprocess_input(aug_img)
        aug_img = tf.cast(aug_img, tf.float32)

        # extract transformed bboxes
        bboxes_transformed = []
        for xmin, ymin, xmax, ymax, _ in transformed['bboxes']:
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            bboxes_transformed.append(tf.convert_to_tensor([cx, cy, w, h], tf.float32))
        bboxes_transformed = tf.convert_to_tensor(bboxes_transformed, tf.float32)
        labels = tf.convert_to_tensor(labels, tf.float32)
        labels = label_encoder.encode_sample(aug_img.shape, bboxes_transformed, labels)
        return [aug_img, labels]

    return preprocess_image


def DetectionGenerator(images_info: dict, label_encoder, batch_size=10):
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
        aug_image, labels = tf.numpy_function(func=detect_augmentation(label_encoder, training),
                                              inp=[image_file, y[0], y[1], y[2]],
                                              Tout=[tf.float32, tf.float32])
        return aug_image, labels

    ds = tf.data.Dataset.zip((image_files_slices, y_slices))
    ds = ds.shuffle(1024)
    n_train = int(0.9 * len(image_files))
    train_ds = ds.take(n_train)
    val_ds = ds.skip(n_train)
    train_ds = train_ds.shuffle(512, reshuffle_each_iteration=True)
    train_ds = train_ds.map(lambda x, y: process_data(x, y, training=True), num_parallel_calls=AUTOTUNE).batch(
        batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: process_data(x, y, training=False),
                        num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return train_ds, val_ds
