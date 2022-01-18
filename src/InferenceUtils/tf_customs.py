import tensorflow as tf
import tensorflow.keras.losses


class Linear(tensorflow.keras.layers.Layer):
    def __init__(self, units=32, input_shape=None, name=None, **kwargs):
        super(Linear, self).__init__(kwargs, name=name)
        self.op_shape = input_shape
        self.units = units
        self.b = None

    def build(self, input_shape):
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            # initializer="random_normal",
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            trainable=True
        )

    def get_config(self):
        # config={}
        config = super(Linear, self).get_config()
        config['units'] = self.units
        config['input_shape'] = self.op_shape
        return dict(list(config.items()))

    def call(self, inputs):
        return tensorflow.keras.layers.LeakyReLU(alpha=0.01)(
            inputs + tf.broadcast_to(self.b, [self.op_shape[0], self.op_shape[1], self.units]))
