"""face_mask dataset."""
from pathlib import Path

import pandas as pd
import tensorflow_datasets as tfds

# TODO(face_mask): Markdown description  that will appear on the catalog page.
from data_utils.mask_generator import create_image_info

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(face_mask): BibTeX citation
_CITATION = """
"""


class FaceMask(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for face_mask dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, **kwargs):
        super(FaceMask, self).__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(face_mask): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
                'labels': tfds.features.Sequence(tfds.features.ClassLabel(num_classes=4)),
                'bboxes': tfds.features.Sequence(tfds.features.BBoxFeature())
            }),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(face_mask): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(Path('train.csv')),
            'val': self._generate_examples(Path('val.csv')),
        }

    def _generate_examples(self, csv_path):
        """Yields examples."""
        # TODO(face_mask): Yields (key, example) tuples from the dataset
        images_info = create_image_info(Path("data/kaggle_mask"), Path("data/medical_mask"))
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
                'image': filename,
                'bboxes': bboxes,
                'labels': labels
            }
