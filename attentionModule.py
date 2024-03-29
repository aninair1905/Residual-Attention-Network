import tensorflow as tf
from keras.layers.convolutional import UpSampling2D

from BaseLayerFile import ResidualBlock


class AttentionModule(object):

    def __init__(self):
        self.p = 1
        self.t = 2
        self.r = 3

        self.residual_block = ResidualBlock()

    def f_prop(self, input, input_channels, scope="attention_module", is_training=True):
        with tf.variable_scope(scope):

            with tf.variable_scope("first_residual_blocks"):
                for i in range(self.p):
                    input = self.residual_block.forward_prop(input, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.variable_scope("trunk_branch"):
                output_trunk = input
                for i in range(self.t):
                    output_trunk = self.residual_block.forward_prop(output_trunk, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.variable_scope("soft_mask_branch"):

                with tf.variable_scope("down_sampling_1"):
                    # max pooling
                    filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(input, ksize=filter_, strides=filter_, padding='SAME')

                    for i in range(self.r):
                        output_soft_mask = self.residual_block.forward_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                with tf.variable_scope("skip_connection"):
                    output_skip_connection = self.residual_block.forward_prop(output_soft_mask, input_channels, is_training=is_training)


                with tf.variable_scope("down_sampling_2"):
                    # max pooling
                    filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME')

                    for i in range(self.r):
                        output_soft_mask = self.residual_block.forward_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                with tf.variable_scope("up_sampling_1"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.forward_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

                # add skip connection
                output_soft_mask += output_skip_connection

                with tf.variable_scope("up_sampling_2"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.forward_prop(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)


                with tf.variable_scope("output"):
                    output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)
                    output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)

                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk

            with tf.variable_scope("last_residual_blocks"):
                for i in range(self.p):
                    output = self.residual_block.forward_prop(output, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            return output