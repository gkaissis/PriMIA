from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.keras.applications import vgg16
import tensorflow_federated as tff

_VGG16_KEY = "vgg16"

# Model hparams
flags.DEFINE_enum("model", default=_VGG16_KEY, enum_values=[_VGG16_KEY],
    help="Which model to use. Must have a model_fn in model_helpers.py.")
flags.DEFINE_integer("hidden_units", default=512,
    help="Number of hidden units in penultimate later.")
flags.DEFINE_float("dropout", default=.25,
    help="Dropout for penultimate layer.")
flags.DEFINE_bool("freeze_model", default=True,
    help="Freeze early layers in the model (if its builder fn allows)")

FLAGS = flags.FLAGS


def build_vgg16(freeze, img_width, img_height):
  base_model = vgg16.VGG16(
      include_top=False,
      input_shape=(img_width, img_height, 3),
      pooling='avg')
  if freeze:
    for layer in base_model.layers:
      layer.trainable=False
  model = tf.keras.models.Sequential()
  model.add(base_model)
  model.add(tf.keras.layers.Dense(FLAGS.hidden_units, activation="relu"))
  if FLAGS.dropout > 0.0:
    model.add(tf.keras.layers.Dropout(FLAGS.dropout))
  model.add(tf.keras.layers.Dense(3))
  return model

def vgg16_fn():
  return build_vgg16(
      freeze=FLAGS.freeze_model,
      img_width=FLAGS.image_width,
      img_height=FLAGS.image_height)

#################

_model_fn_factory = {
    _VGG16_KEY: vgg16_fn,
}

def model_fn_factory(model_name):
  return _model_fn_factory[model_name]

if __name__ == '__main__':
  app.run(lambda _: None)
