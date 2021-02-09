import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class T2PGenerator(keras.Model):
    def __init__(self):
        super(T2PGenerator, self).__init__()

        activation = tf.nn.relu

        self.fc_1 = Dense(512, activation=activation)
        self.fc_2 = Dense(256, activation=activation)
        self.fc_3 = Dense(128, activation=activation)
        self.fc_4 = Dense(3, activation=tf.nn.sigmoid)

    def call(self, z, context):
        inputs = tf.concat((z, context), 1)

        x = self.fc_1(inputs)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)

        return x        # (batch, 3)


class T2PDiscriminator(keras.Model):
    def __init__(self):
        super(T2PDiscriminator, self).__init__()
        activation = tf.nn.leaky_relu

        self.fc_1 = Dense(512, activation=activation)
        self.fc_2 = Dense(256, activation=activation)
        self.fc_3 = Dense(128, activation=activation)
        self.fc_4 = Dense(1, activation=tf.nn.sigmoid)

    def call(self, result, context):
        inputs = tf.concat([result, context], 1)

        x = self.fc_1(inputs)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)

        return x        # (batch ,1)
