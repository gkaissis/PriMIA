from absl import flags
import tensorflow as tf

# Device environment flags
flags.DEFINE_bool('use_gpu', default=True,
    help='use GPU if available')
flags.DEFINE_integer('gpu_id', default=0,
    help='which (physical) GPU to use if use_gpus was set')
flags.DEFINE_integer('memory_per_device', default=512,
    help='amount of memory to limit each logical device to. this should equal '
         'total_memory / num_gpu_devices.')

FLAGS = flags.FLAGS

def configure_client_gpus(num_devices):
  physical_devices = tf.config.list_physical_devices('GPU')
  physical_devices = [dv for dv in physical_devices
      if dv.name.split(":")[-1] == FLAGS.gpu_id]
  tf.config.set_visible_devices(physical_devices, 'GPU')
  gpu_device = physical_devices[0]
  configs = [tf.config.LogicalDeviceConfiguration(
      memory_limit=FLAGS.memory_per_device) for _ in range(num_devices)]
  tf.config.set_logical_device_configuration(gpu_device, configs)
