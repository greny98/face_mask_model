"""coco_ds dataset."""
import tensorflow_datasets as tfds
import fiftyone.zoo as foz

# TODO(coco_ds): Markdown description  that will appear on the catalog page.
from configs.coco_configs import coco_name2idx

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(coco_ds): BibTeX citation
_CITATION = """
"""


class CocoDs(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for coco_ds dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name2idx = coco_name2idx

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(coco_ds): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
                'labels': tfds.features.Sequence(tfds.features.ClassLabel(num_classes=80)),
                'bboxes': tfds.features.Sequence(tfds.features.BBoxFeature())
            }),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        train_ds = foz.load_zoo_dataset(name="coco-2017", split="train", label_types=["detections"])
        val_ds = foz.load_zoo_dataset(name="coco-2017", split="validation", label_types=["detections"])
        return {
            'train': self._generate_examples(train_ds),
            'val': self._generate_examples(val_ds),
        }

    def _generate_examples(self, ds):
        """Yields examples."""
        # TODO(coco_ds): Yields (key, example) tuples from the dataset
        for sample in ds:
            if len(sample.ground_truth.detections) == 0:
                continue
            labels = []
            bboxes = []
            for detection in sample.ground_truth.detections:
                x, y, w, h = detection.bounding_box
                bboxes.append(
                    tfds.features.BBox(
                        ymin=min(y, 1.), xmin=min(x, 1.),
                        ymax=min(y + h, 1.), xmax=min(x + w, 1.)))
                labels.append(self._name2idx[detection.label])
            yield sample.id, {
                'image': sample.filepath,
                'labels': labels,
                'bboxes': bboxes
            }
