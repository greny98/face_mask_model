import argparse
import data_utils.fire_ds
from tensorflow.keras import optimizers, callbacks
from data_utils.data_generator_large import DetectionGenerator
from model.anchor_boxes_large import LabelEncoderLarge
from model.losses import RetinaNetLoss
from model.ssd import create_ssd_model, transfer_ssd
import tensorflow_datasets as tfds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='ckpt')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--coco_ckpt', type=str, default='ckpt/coco_large/checkpoint')
    args = vars(parser.parse_args())
    return args


def schedule(e, lr):
    if e < 12 or e % 6 != 0:
        return lr
    return 0.925 * lr


if __name__ == '__main__':
    args = parse_args()
    print(args)
    label_encoder = LabelEncoderLarge()
    train_ds, val_ds = tfds.load('fire_ds', split=['train', 'val'], shuffle_files=True)
    train_ds = DetectionGenerator(train_ds, label_encoder, batch_size=args["batch_size"], training=True)
    val_ds = DetectionGenerator(val_ds, label_encoder, batch_size=args["batch_size"], training=False)
    # Create Model
    num_classes = 1
    ssd_model = transfer_ssd(args["coco_ckpt"], num_classes, large=True)
    # ssd_model.summary()
    loss_fn = RetinaNetLoss(num_classes)
    ssd_model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(learning_rate=args['lr']))
    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=f"{args['output_dir']}/checkpoint",
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss')
    lr_schedule_cb = callbacks.LearningRateScheduler(schedule)
    # Tensorboard
    tensorboard_cb = callbacks.TensorBoard(log_dir=args['log_dir'])
    ssd_model.fit(train_ds,
                  validation_data=val_ds,
                  callbacks=[ckpt_cb, lr_schedule_cb, tensorboard_cb],
                  epochs=args['epochs'])
