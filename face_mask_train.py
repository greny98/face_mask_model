import argparse
import data_utils.face_mask
from tensorflow.keras import optimizers, callbacks
from data_utils.mask_generator import DetectionGenerator
from model.anchor_boxes_large import LabelEncoderLarge
from model.losses import RetinaNetLoss
from model.ssd import create_face_mask_model
import tensorflow_datasets as tfds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='ckpt/face_mask')
    parser.add_argument('--coco_ckpt', type=str, default='ckpt/coco/checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = vars(parser.parse_args())
    return args


def schedule(e, lr):
    if e < 7 or e % 4 != 0:
        return lr
    return 0.925 * lr


if __name__ == '__main__':
    args = parse_args()
    print(args)
    label_encoder = LabelEncoderLarge()
    train_ds, val_ds = tfds.load('face_mask', split=['train', 'val'], shuffle_files=True)
    train_ds = DetectionGenerator(train_ds, label_encoder, batch_size=args["batch_size"], training=True)
    val_ds = DetectionGenerator(val_ds, label_encoder, batch_size=args["batch_size"], training=False)
    # Create Model
    num_classes = 4
    ssd_model = create_face_mask_model(args['coco_ckpt'])
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
