import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations

from ocr.resnet import create_resnet_stage


# pylint: disable=abstract-method
# pylint: disable=too-many-instance-attributes
class ImageEncoder(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.resnet_stage_1 = create_resnet_stage(1)
        self.resnet_stage_2 = create_resnet_stage(2)
        self.resnet_stage_3 = create_resnet_stage(3)
        self.resnet_stage_4 = create_resnet_stage(4)
        self.resnet_stage_5 = create_resnet_stage(5)

        self.to_256ch_2 = layers.Conv2D(256, (1, 1), padding='same', name='to_256ch_2')
        self.to_256ch_3 = layers.Conv2D(256, (1, 1), padding='same', name='to_256ch_3')
        self.to_256ch_4 = layers.Conv2D(256, (1, 1), padding='same', name='to_256ch_4')
        self.to_256ch_5 = layers.Conv2D(256, (1, 1), padding='same', name='to_256ch_5')

        self.up5_to_out = layers.UpSampling2D(size=(8, 8), name='up5_to_out')
        self.up5_to_in4 = layers.UpSampling2D(size=(2, 2), name='up5_to_in4')
        self.up4_to_out = layers.UpSampling2D(size=(4, 4), name='up4_to_out')
        self.up4_to_in3 = layers.UpSampling2D(size=(2, 2), name='up4_to_in3')
        self.up3_to_out = layers.UpSampling2D(size=(2, 2), name='up3_to_out')
        self.up3_to_in2 = layers.UpSampling2D(size=(2, 2), name='up3_to_in2')

        self.conv_out5 = layers.Conv2D(64, (3, 3), padding='same', name='conv_p5')
        self.conv_out4 = layers.Conv2D(64, (3, 3), padding='same', name='conv_p4')
        self.conv_out3 = layers.Conv2D(64, (3, 3), padding='same', name='conv_p3')
        self.conv_out2 = layers.Conv2D(64, (3, 3), padding='same', name='conv_p2')

    # pylint: disable=too-many-locals
    def call(self, inputs, training=False, mask=None):
        img = inputs  # batch x h x w x channels
        stage1_out = self.resnet_stage_1(img) # h/4 x w/4 x 64
        stage2_out = self.resnet_stage_2(stage1_out)  # h/4 x w/4 x 256
        stage3_out = self.resnet_stage_3(stage2_out)  # h/8 x w/8 x 512
        stage4_out = self.resnet_stage_4(stage3_out)  # h/16 x w/16 x 1024
        stage5_out = self.resnet_stage_5(stage4_out)  # h/32 x w/32 x 2048

        in2 = self.to_256ch_2(stage2_out)  # h/4 x w/4 x 256
        in3 = self.to_256ch_3(stage3_out)  # h/8 x w/8 x 256
        in4 = self.to_256ch_4(stage4_out)  # h/16 x w/16 x 256
        in5 = self.to_256ch_5(stage5_out)  # h/32 x w/32 x 256

        out5 = self.up5_to_out(self.conv_out5(in5))  # 1/32 * 8 = 1/4
        in4 = layers.Add()([  # 1/16
            in4,  # 1/16
            self.up5_to_in4(in5),  # 1/32 * 2 = 1/16
        ])

        out4 = self.up4_to_out(self.conv_out4(in4))  # 1/16 * 4 = 1/4
        in3 = layers.Add()([  # 1/8
            in3,  # 1/8
            self.up4_to_in3(in4), # 1/16 * 2 = 1/8
        ])

        out3 = self.up3_to_out(self.conv_out3(in3))  # 1/8 * 2 = 1/4
        in2 = layers.Add()([  # 1/4
            in2,  # 1/4
            self.up3_to_in2(in3),  # 1/8 * 2 = 1/4
        ])

        out2 = self.conv_out2(in2)

        out = layers.Concatenate()([out2, out3, out4, out5])

        return out


class FeatureToImage(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.conv = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)
        self.bn_1 = layers.BatchNormalization()
        self.deconv_1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), use_bias=False)
        self.bn_2 = layers.BatchNormalization()
        self.deconv_2 = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2))

    def call(self, inputs, training=False, mask=None):
        x = inputs
        x = self.conv(x)
        x = self.bn_1(x)
        x = activations.relu(x)

        x = self.deconv_1(x)
        x = self.bn_2(x)
        x = activations.relu(x)

        x = self.deconv_2(x)
        x = activations.sigmoid(x)

        return x
