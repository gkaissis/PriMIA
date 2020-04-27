import tensorflow as tf
from tensorflow.keras.applications import vgg16


def compile_local_model(model, optimizer="adam"):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer, [loss_fn], [acc_fn])


def build_vgg16(freeze=True):
    base_model = vgg16.VGG16(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    if freeze:
        for layer in base_model.layers:
            layer.trainable=False
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(tf.keras.layers.Dense(3))
    return model

