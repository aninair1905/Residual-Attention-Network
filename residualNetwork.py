import numpy as np
import tensorflow as tf

from BaseLayerFile import ResidualBlock
from attentionModule import AttentionModule


class ResidualAttentionNetwork(object):

    def __init__(self):
        self.input_shape = 10
        self.output_dim = [-1,32,32,3]

        self.attention_module = AttentionModule()
        self.residual_block = ResidualBlock()

    def f_prop(self, x, is_training=True):

        x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='SAME')

        x = tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')

        x = self.attention_module.f_prop(x, input_channels=32, scope="attention_module_1", is_training=is_training)

        x = self.residual_block.forward_prop(x, input_channels=32, output_channels=64, scope="residual_block_1",is_training=is_training)

        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


        x = self.attention_module.f_prop(x, input_channels=64, scope="attention_module_2", is_training=is_training)

        x = self.residual_block.forward_prop(x, input_channels=64, output_channels=128, scope="residual_block_2",is_training=is_training)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # attention module, x -> [None, row/2, line/2, 64]
        x = self.attention_module.f_prop(x, input_channels=128, scope="attention_module_3", is_training=is_training)

        # residual block, x-> [None, row/2, line/2, 256]
        x = self.residual_block.forward_prop(x, input_channels=128, output_channels=256, scope="residual_block_3", is_training=is_training)

        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.forward_prop(x, input_channels=256, output_channels=256, scope="residual_block_4", is_training=is_training)

        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.forward_prop(x, input_channels=256, output_channels=256, scope="residual_block_5",
                                       is_training=is_training)

        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.forward_prop(x, input_channels=256, output_channels=256, scope="residual_block_6",is_training=is_training)

        x = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)

        return y
