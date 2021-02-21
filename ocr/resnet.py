from typing import Tuple

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations


def create_resnet_stage_1(
    filters_div: int,
    input_ch: int) -> tf.keras.Model:

    return tf.keras.Sequential(
        layers=[
            layers.Input(shape=(None, None, input_ch), name='image_input'),
            layers.ZeroPadding2D((3, 3), name='conv1_pad'),
            layers.Conv2D(
                int(64 / filters_div),
                (7, 7),
                strides=(2, 2),
                padding='valid',
                use_bias=False,
                name='conv1'),
            layers.BatchNormalization(axis=3, name='bn_conv1'),
            layers.Activation('relu'),
            layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        ],
        name='stage1',
    )


def create_resnet_stage_2(filters_div: int):
    filters = (
        int(64 / filters_div),
        int(64 / filters_div),
        int(256 / filters_div))
    return tf.keras.Sequential(
        layers=[
            layers.Input(
                shape=(None, None, int(64 / filters_div)),
                name='stage2_input'),
            ConvBlock(
                kernel_size=3, filters=filters, stage=(2, 'a'), stride=1),
            IdentityBlock(
                kernel_size=3, filters=filters, stage=(2, 'b')),
            IdentityBlock(
                kernel_size=3, filters=filters, stage=(2, 'c')),
        ],
        name='stage2',
    )


def create_resnet_stage_3(filters_div: int):
    filters = (
        int(128 / filters_div),
        int(128 / filters_div),
        int(512 / filters_div))
    return tf.keras.Sequential(
        layers=[
            layers.Input(
                shape=(None, None, int(256 / filters_div)),
                name='stage3_input'),
            ConvBlock(kernel_size=3, filters=filters, stage=(3, 'a')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(3, 'b')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(3, 'c')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(3, 'd')),
        ],
        name='stage3')


def create_resnet_stage_4(filters_div: int):
    filters = (
        int(256 / filters_div),
        int(256 / filters_div),
        int(1024 / filters_div))
    return tf.keras.Sequential(
        layers=[
            layers.Input(
                shape=(None, None, int(512 / filters_div)),
                name='stage4_input'),
            ConvBlock(kernel_size=3, filters=filters, stage=(4, 'a')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(4, 'b')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(4, 'c')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(4, 'd')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(4, 'e')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(4, 'f')),
        ],
        name='stage4')


def create_resnet_stage_5(filters_div: int):
    filters = (
        int(512 / filters_div),
        int(512 / filters_div),
        int(2048 / filters_div))
    return tf.keras.Sequential(
        layers=[
            layers.Input(
                shape=(None, None, int(1024 / filters_div)),
                name='stage5_input'),
            ConvBlock(kernel_size=3, filters=filters, stage=(5, 'a')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(5, 'b')),
            IdentityBlock(kernel_size=3, filters=filters, stage=(5, 'c')),
        ],
        name='stage5')


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
            use_bias=False,
            name=conv_name_base + '2a',
        )
        self.bn1 = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')

        self.c_2 = layers.Conv2D(
            filters=f_2,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding='same',
            use_bias=False,
            name=conv_name_base + '2b')
        self.bn2 = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')

        self.c_3 = layers.Conv2D(
            filters=f_3,
            kernel_size=(1, 1),
            use_bias=False,
            name=conv_name_base + '2c')
        self.bn3 = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')

        self.c_4 = layers.Conv2D(
            filters=f_3,
            kernel_size=(1, 1),
            strides=(stride, stride),
            use_bias=False,
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
            use_bias=False,
            name=conv_name_base + '2a')
        self.bn1 = layers.BatchNormalization(
            axis=3,
            name=bn_name_base + '2a')

        self.c_2 = layers.Conv2D(
            filters=f_2,
            kernel_size=(kernel_size, kernel_size),
            padding='same',
            use_bias=False,
            name=conv_name_base + '2b')
        self.bn2 = layers.BatchNormalization(
            axis=3,
            name=bn_name_base + '2b')

        self.c_3 = layers.Conv2D(
            filters=f_3,
            kernel_size=(1, 1),
            use_bias=False,
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
