from os.path import join
import numpy as np
import tensorflow as tf

from data_utils.kaggle_mask import read_kaggle_mask
from data_utils.medical_mask import read_medical_mask

import albumentations as augment
from configs.common_config import IMAGE_SIZE

AUTOTUNE = tf.data.AUTOTUNE


def create_image_info(kaggle_dir, medical_dir):
    info = read_kaggle_mask(join(kaggle_dir, 'annotations'), join(kaggle_dir, 'images'), {})
    info = read_medical_mask(join(medical_dir, 'annotations'), join(medical_dir, 'images'), info)
    return info


def detect_augmentation(label_encoder, training):
    if training:
        transform = augment.Compose([
            augment.LongestMaxSize(224),
            augment.ImageCompression(quality_lower=70, quality_upper=100),
            augment.ChannelShuffle(),
            augment.HorizontalFlip(p=0.3),
            augment.RandomBrightnessContrast(0.3, 0.3, p=0.4),
            augment.Rotate(10),
            augment.GaussNoise(p=0.4),
            augment.GaussianBlur(p=0.4),
            augment.RandomSizedBBoxSafeCrop(IMAGE_SIZE, IMAGE_SIZE, p=0.5),
            augment.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ], bbox_params=augment.BboxParams(format='pascal_voc'))
    else:
        transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)],
                                    bbox_params=augment.BboxParams(format='pascal_voc'))

    def augmentation(image, bboxes, labels):
        if bboxes.shape[0] == 0:
            print("\n===========")
            print("Empty BBox!!!")
            print("===========\n")
        img_h, img_w, _ = image.shape
        # Hande bboxes
        ymin = bboxes[:, [0]] * img_h
        xmin = bboxes[:, [1]] * img_w
        ymax = bboxes[:, [2]] * img_h
        xmax = bboxes[:, [3]] * img_w

        extend_boxes = np.ones(shape=(bboxes.shape[0], 1))
        bboxes = np.concatenate([xmin, ymin, xmax, ymax, extend_boxes], axis=-1)
        mask = (bboxes[:, 3] > bboxes[:, 1]) & (bboxes[:, 2] > bboxes[:, 0])
        bboxes = bboxes[mask]
        labels = labels[mask]
        # Add info
        data = {'image': image, 'bboxes': bboxes}
        transformed = transform(**data)
        # extract transformed image
        aug_img = transformed['image']
        aug_img = tf.cast(aug_img, tf.float32)
        aug_img = aug_img / 127.5 - 1
        # extract boxes
        bboxes_transformed = []
        for xmin, ymin, xmax, ymax, _ in transformed['bboxes']:
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            w = xmax - xmin
            h = ymax - ymin
            bboxes_transformed.append(tf.convert_to_tensor([cx, cy, w, h], tf.float32))

        bboxes_transformed = tf.convert_to_tensor(bboxes_transformed, tf.float32)
        labels = tf.convert_to_tensor(labels, tf.float32)
        labels = label_encoder.encode_sample(aug_img.shape, bboxes_transformed, labels)

        # return [aug_img, labels]
        return [aug_img, bboxes_transformed]

    return augmentation


def DetectionGenerator(datasets, label_encoder, batch_size, training):
    def preprocess_image(info):
        image = info["image"]
        bboxes = info["bboxes"]
        labels = info["labels"]
        aug_image, labels = tf.numpy_function(func=detect_augmentation(label_encoder, training),
                                              inp=[image, bboxes, labels],
                                              Tout=[tf.float32, tf.float32])
        return aug_image, labels

    datasets = (datasets.shuffle(600, reshuffle_each_iteration=training)
                .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    return datasets
