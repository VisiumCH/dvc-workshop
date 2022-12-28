import tensorflow as tf


class EfficentNet(tf.keras.Model):
    def __init__(self, image_height, image_width, channels, pooling, include_top, activation, class_indices):
        super(EfficentNet, self).__init__()
        self.class_indices = class_indices
        self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=include_top,
            pooling=pooling,
            input_shape=(image_height, image_width, channels),
        )
        self.set_trainable(False)
        self.output_layer = tf.keras.layers.Dense(len(class_indices), activation=activation)

    def call(self, inputs):
        x = self.base_model(inputs)
        return self.output_layer(x)

    def set_trainable(self, trainable=True):
        self.base_model.trainable = trainable
