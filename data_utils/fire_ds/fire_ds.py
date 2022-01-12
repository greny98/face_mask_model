"""fire_ds dataset."""
import os
from os.path import join
from pathlib import Path

import pandas as pd
import tensorflow_datasets as tfds

# TODO(fire_ds): Markdown description  that will appear on the catalog page.
from data_utils.pascal import xml_to_dict

_DESCRIPTION = """
"""

# TODO(fire_ds): BibTeX citation
_CITATION = """
"""


def read_fire(annotation, target: dict) -> dict:
    FIRE = {"fire": 0}
    for xml_file in os.listdir(annotation):
        if ".xml" not in xml_file:
            continue
        anno = xml_to_dict(join(annotation, xml_file))
        image_info = {
            "bboxes": [],
            "labels": [],
            'width': anno["size"]["width"],
            'height': anno["size"]["height"]}
        if anno["size"]["width"] * anno["size"]["height"] <= 0:
            continue
        for obj in anno["object"]:
            if obj["name"].lower() not in FIRE.keys():
                continue
            bbox = obj["bndbox"]
            image_info["bboxes"].append([
                bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
            ])
            image_info["labels"].append(FIRE[obj["name"].lower()])
        if len(image_info["labels"]) > 0:
            target[anno["filename"]] = image_info
        else:
            print(xml_file)
    return target


class FireDs(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for fire_ds dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(fire_ds): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
                'labels': tfds.features.Sequence(tfds.features.ClassLabel(num_classes=1)),
                'bboxes': tfds.features.Sequence(tfds.features.BBoxFeature())
            }),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {
            'train': self._generate_examples(Path('fire_train.csv')),
            'val': self._generate_examples(Path('fire_val.csv')),
            'test': self._generate_examples(Path('fire_test.csv')),
        }

    def _generate_examples(self, csv_path):
        """Yields examples."""
        # TODO(fire_ds): Yields (key, example) tuples from the dataset
        images_info = read_fire(Path("data/fire/annotations"), {})
        df = pd.read_csv(csv_path)

        for filename, width, height in df[['filename', 'width', 'height']].values:
            info = images_info.get(filename, None)
            if info is None:
                continue
            bboxes = []
            labels = []
            for bbox, label in zip(info['bboxes'], info['labels']):
                xmin, ymin, xmax, ymax = bbox
                bboxes.append(
                    tfds.features.BBox(
                        ymin=min(ymin / height, 1.), xmin=min(xmin / width, 1.),
                        ymax=min(ymax / height, 1.), xmax=min(xmax / width, 1.)))
                labels.append(label)
            yield filename, {
                'image': Path("data/fire/images") / filename,
                'bboxes': bboxes,
                'labels': labels
            }
