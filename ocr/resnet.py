from typing import List

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
                ConvBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block='a', stride=1),
                IdentityBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block='b'),
                IdentityBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block='c'),
            ],
            name='stage2'
        )
    if idx == 3:
        return tf.keras.Sequential(
            layers=[
                layers.Input(shape=(None, None, 256), name='stage3_input'),
                ConvBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block='a', stride=2),
                IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block='b'),
                IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block='c'),
                IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block='d'),
            ],
            name='stage3',
        )
    if idx == 4:
        return tf.keras.Sequential(
            layers=[
                layers.Input(shape=(None, None, 512), name='stage4_input'),
                ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', stride=2),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='b'),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='c'),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='d'),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='e'),
                IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block='f'),
            ],
            name='stage4',
        )
    if idx == 5:
        return tf.keras.Sequential(
            layers=[
                layers.Input(shape=(None, None, 1024), name='stage5_input'),
                ConvBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block='a', stride=2),
                IdentityBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block='b'),
                IdentityBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block='c'),
            ],
            name='stage5',
        )
    
    
class ConvBlock(tf.keras.Model):
    
    def __init__(self, kernel_size: int, filters: List[int], stage: int, block: str, stride=2, **kwargs):
        super().__init__(kwargs)
        conv_name_base = 'res{}{}_branch'.format(stage, block)
        bn_name_base = 'bn{}{}_branch'.format(stage, block)

        F1, F2, F3 = filters

        self.c1 = layers.Conv2D(
            filters=F1,
            kernel_size=(1, 1),
            strides=(stride, stride),
            padding='valid',
            name=conv_name_base + '2a',
        )
        self.bn1 = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')

        self.c2 = layers.Conv2D(
            filters=F2,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='same',
            name=conv_name_base + '2b')
        self.bn2 = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')

        self.c3 = layers.Conv2D(
            filters=F3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2c')
        self.bn3 = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')

        self.c4 = layers.Conv2D(
            filters=F3,
            kernel_size=(1, 1),
            strides=(stride, stride),
            padding='valid',
            name=conv_name_base + '1')
        self.bn4 = layers.BatchNormalization(axis=3, name=bn_name_base + '1')

    
    def call(self, inputs):
        x = inputs
        x_shortcut = x
        
        x = self.c1(x)
        x = self.bn1(x)
        x = activations.relu(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = activations.relu(x)
        
        x = self.c3(x)
        x = self.bn3(x)
        
        x_shortcut = self.c4(x_shortcut)
        x_shortcut = self.bn4(x_shortcut)
        
        x = layers.Add()([x, x_shortcut])
        x = activations.relu(x)
        
        return x


class IdentityBlock(tf.keras.Model):
    
    def __init__(self, kernel_size: int, filters: List[int], stage: int, block: str, stride=2, **kwargs):
        super().__init__(kwargs)
        conv_name_base = 'res{}{}_branch'.format(stage, block)
        bn_name_base = 'bn{}{}_branch'.format(stage, block)
        f1, f2, f3 = filters
        
        self.c1 = layers.Conv2D(
            filters=f1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2a')
        self.bn1 = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')
        
        self.c2 = layers.Conv2D(
            filters=f2,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='same',
            name=conv_name_base + '2b')
        self.bn2 = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')
        
        self.c3 = layers.Conv2D(
            filters=f3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name=conv_name_base + '2c')
        self.bn3 = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')        
        
    def call(self, inputs):
        x = inputs
        x_shortcut = x
        
        x = self.c1(x)
        x = self.bn1(x)
        x = activations.relu(x)
        
        x = self.c2(x)
        x = self.bn2(x)
        x = activations.relu(x)
        
        x = self.c3(x)
        x = self.bn3(x)
        
        x = layers.Add()([x, x_shortcut])
        x = activations.relu(x)
        
        return x