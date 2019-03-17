
"""
Base Models for he RAN
"""


import tensorflow as tf

class Layers(object):

    def __init__(self, shape):
        self.W = self.weight_variable(shape)
        self.b = tf.Variable(tf.zeros([shape[1]]))

    @staticmethod
    def weight_variable(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def forward_prop(self, x):
        return tf.matmul(x, self.W)+ self.b

class Dense(Layers):
    def __init__(self, shape, function=tf.nn.softmax):
        super().__init__(shape)
        self.function = function

    def forward_prop(self, x):
        return self.function(tf.matmul(self.W + x) + self.b)

class ResidualBlock(object):

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def forward_prop(self, input, input_channels, output_channels=None, scope="residual_block", is_training=True):
        if output_channels is None:
            output_channels = input_channels

        with tf.variable_scope(scope):

            x = self.batch_norm(input, input_channels, is_training)

            x = tf.layers.conv2d(x, filters=output_channels, kernel_size=1, padding='SAME', name="conv1")

            x = self.batch_norm(x, output_channels, is_training)

            x = tf.layers.conv2d(x, filters=output_channels, kernel_size=self.kernel_size,
                                 strides=1, padding='SAME', name="conv2")


            if input_channels != output_channels:
                _input = tf.layers.conv2d(input, filters=output_channels, kernel_size=1, strides=1)

            output = x + input

            return output

    def batch_norm(x, n_out, is_training=True):
        with tf.variable_scope('batch_norm'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(tf.cast(is_training, tf.bool),
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return tf.nn.relu(normed)

