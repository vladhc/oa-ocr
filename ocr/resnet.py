from typing import Tuple

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations


def create_resnet_stage(idx: int) -> tf.keras.Model:

    if idx == 1:
        return tf.keras.Sequential(
            layers=[
                layers.Input(shape=(None, None, 3), name='image_input'),
                layers.ZeroPadding2D((3, 3)),
                layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1'),
                layers.BatchNormalization(axis=3, name='bn_conv1'),
                layers.Activation('relu'),
                layers.ZeroPadding2D((1, 1)),
                layers.MaxPooling2D((3, 3), strides=(2, 2)),
            ],
            name='stage1',
        )
    if idx == 2:
        return tf.keras.Sequential(
            layers=[
                layers.Input(shape=(None, None, 64), name='stage2_input'),
                ConvBlock(
                    kernel_size=3, filters=(64, 64, 256), stage=(2, 'a'), stride=1),
                IdentityBlock(
                    kernel_size=3, filters=(64, 64, 256), stage=(2, 'b')),
                IdentityBlock(kernel_size=3, filters=(64, 64, 256), stage=(2, 'c')),
            ],
            name='stage2'
        )
    if idx == 3:
        return tf.keras.Sequential(
            layers=[
                layers.Input(shape=(None, None, 256), name='stage3_input'),
                ConvBlock(kernel_size=3, filters=(128, 128, 512), stage=(3, 'a'), stride=2),
                IdentityBlock(kernel_size=3, filters=(128, 128, 512), stage=(3, 'b')),
                IdentityBlock(kernel_size=3, filters=(128, 128, 512), stage=(3, 'c')),
                IdentityBlock(kernel_size=3, filters=(128, 128, 512), stage=(3, 'd')),
            ],
            name='stage3',
        )
    if idx == 4:
        return tf.keras.Sequential(
            layers=[
                layers.Input(shape=(None, None, 512), name='stage4_input'),
                ConvBlock(kernel_size=3, filters=(256, 256, 1024), stage=(4, 'a'), stride=2),
                IdentityBlock(kernel_size=3, filters=(256, 256, 1024), stage=(4, 'b')),
                IdentityBlock(kernel_size=3, filters=(256, 256, 1024), stage=(4, 'c')),
                IdentityBlock(kernel_size=3, filters=(256, 256, 1024), stage=(4, 'd')),
                IdentityBlock(kernel_size=3, filters=(256, 256, 1024), stage=(4, 'e')),
                IdentityBlock(kernel_size=3, filters=(256, 256, 1024), stage=(4, 'f')),
            ],
            name='stage4',
        )
    if idx == 5:
        return tf.keras.Sequential(
            layers=[
                layers.Input(shape=(None, None, 1024), name='stage5_input'),
                ConvBlock(kernel_size=3, filters=(512, 512, 2048), stage=(5, 'a'), stride=2),
                IdentityBlock(kernel_size=3, filters=(512, 512, 2048), stage=(5, 'b')),
                IdentityBlock(kernel_size=3, filters=(512, 512, 2048), stage=(5, 'c')),
            ],
            name='stage5',
        )
    raise ValueError("ResNet stage index should be in the range 1..5")


# pylint: disable=abstract-method
# pylint: disable=too-many-instance-attributes
class ConvBlock(tf.keras.Model):

    def __init__(
            self,
            kernel_size: int,
            filters: Tuple[int, int, int],
            stage: Tuple[int, str],
            stride=2,
            **kwargs):

        super().__init__(kwargs)
        stage_id, block = stage
        conv_name_base = 'res{}{}_branch'.format(stage_id, block)
        bn_name_base = 'bn{}{}_branch'.format(stage_id, block)

        f_1, f_2, f_3 = filters

        self.c_1 = layers.Conv2D(
            filters=f_1,
            kernel_size=(1, 1),
            strides=(stride, stride),
            padding='valid',
            name=conv_name_base + '2a',
        )
        self.bn1 = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')

        self.c_2 = layers.Conv2D(
            filters=f_2,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='same',
            name=conv_name_base + '2b')
        self.bn2 = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')

        self.c_3 = layers.Conv2D(
            filters=f_3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2c')
        self.bn3 = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')

        self.c_4 = layers.Conv2D(
            filters=f_3,
            kernel_size=(1, 1),
            strides=(stride, stride),
            padding='valid',
            name=conv_name_base + '1')
        self.bn4 = layers.BatchNormalization(axis=3, name=bn_name_base + '1')

    def call(self, inputs, training=False, mask=None):
        x = inputs
        x_shortcut = x

        x = self.c_1(x)
        x = self.bn1(x, training=training)
        x = activations.relu(x)

        x = self.c_2(x)
        x = self.bn2(x, training=training)
        x = activations.relu(x)

        x = self.c_3(x)
        x = self.bn3(x, training=training)

        x_shortcut = self.c_4(x_shortcut)
        x_shortcut = self.bn4(x_shortcut, training=training)

        x = layers.Add()([x, x_shortcut])
        x = activations.relu(x)

        return x


# pylint: disable=abstract-method
# pylint: disable=too-many-instance-attributes
class IdentityBlock(tf.keras.Model):

    def __init__(
            self,
            kernel_size: int,
            filters: Tuple[int, int, int],
            stage: Tuple[int, str],
            **kwargs):

        super().__init__(kwargs)
        stage_id, block = stage
        conv_name_base = 'res{}{}_branch'.format(stage_id, block)
        bn_name_base = 'bn{}{}_branch'.format(stage_id, block)
        f_1, f_2, f_3 = filters

        self.c_1 = layers.Conv2D(
            filters=f_1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2a')
        self.bn1 = layers.BatchNormalization(
            axis=3,
            name=bn_name_base + '2a')

        self.c_2 = layers.Conv2D(
            filters=f_2,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='same',
            name=conv_name_base + '2b')
        self.bn2 = layers.BatchNormalization(
            axis=3,
            name=bn_name_base + '2b')

        self.c_3 = layers.Conv2D(
            filters=f_3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2c')
        self.bn3 = layers.BatchNormalization(
            axis=3,
            name=bn_name_base + '2c')

    def call(self, inputs, training=False, mask=None):
        x = inputs
        x_shortcut = x

        x = self.c_1(x)
        x = self.bn1(x, training=training)
        x = activations.relu(x)

        x = self.c_2(x)
        x = self.bn2(x, training=training)
        x = activations.relu(x)

        x = self.c_3(x)
        x = self.bn3(x, training=training)

        x = layers.Add()([x, x_shortcut])
        x = activations.relu(x)

        return x
