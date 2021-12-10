from tensorflow.keras import layers, Model, regularizers

from configs.pascal_configs import PASCAL_LABELS
from model.feature_pyramid import get_backbone, FeaturePyramid

l2 = regularizers.l2(1.5e-5)


def build_head(feature, name):
    for i in range(4):
        feature = layers.Conv2D(128, 3, padding="same", name=name + '_conv' + str(i))(feature)
        feature = layers.BatchNormalization(epsilon=1.001e-5, name=f'{name}_bn_{i}')(feature)
        feature = layers.ReLU()(feature)
    return feature


def ssd_head(features):
    classes_outs = []
    box_outputs = []
    for idx, feature in enumerate(features):
        classify_head = build_head(feature, 'classify_head' + str(idx))
        detect_head = build_head(feature, 'detect_head' + str(idx))
        box_outputs.append(detect_head)
        classes_outs.append(classify_head)
    return classes_outs, box_outputs


def create_ssd_model(num_classes):
    backbone = get_backbone()
    pyramid = FeaturePyramid(backbone)
    classes_heads, box_heads = ssd_head(pyramid.outputs)

    num_anchor_boxes = 9
    classes_outs = []
    box_outputs = []
    for idx, head in enumerate(box_heads):
        detect_head = layers.Conv2D(num_anchor_boxes * 4, 3, padding="same",
                                    name='detect_head' + str(idx) + '_conv_out',
                                    kernel_regularizer=l2)(head)
        box_outputs.append(layers.Reshape([-1, 4])(detect_head))

    for idx, head in enumerate(classes_heads):
        classify_head = layers.Conv2D(num_anchor_boxes * num_classes, 3, padding="same",
                                      name='classify_head' + str(idx) + '_conv_out',
                                      kernel_regularizer=l2)(head)
        classes_outs.append(layers.Reshape([-1, num_classes])(classify_head))

    classes_outs = layers.Concatenate(axis=1)(classes_outs)
    box_outputs = layers.Concatenate(axis=1)(box_outputs)
    outputs = layers.Concatenate(axis=-1)([box_outputs, classes_outs])
    return Model(inputs=[backbone.input], outputs=[outputs])


def create_face_mask_model(pascal_ckpt):
    coco_model = create_ssd_model(len(PASCAL_LABELS))
    coco_model.load_weights(pascal_ckpt).expect_partial()
    face_mask_model = create_ssd_model(4)
    for l in face_mask_model.layers:
        if ('classify_head' not in l.name) and ('_conv_out' not in l.name) and (len(l.weights) > 0):
            l.set_weights(coco_model.get_layer(l.name).weights)
    return face_mask_model
