from configs.mask_configs import KAGGLE_MASK_LABELS
from data_utils.pascal import read_pascal


def read_kaggle_mask(annotation, images_dir, target: dict):
    return read_pascal(annotation, images_dir, KAGGLE_MASK_LABELS, target)
