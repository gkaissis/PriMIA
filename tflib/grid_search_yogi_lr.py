from absl import app
from absl import flags

from tff_optim.utils import utils_impl

flags.DEFINE_string('flag_file', default='tflib/config/default.cfg',
    help='default params flagfile')
flags.DEFINE_integer('max_workers', default=1,
    help='max number of parallel experiment processes')

FLAGS = flags.FLAGS

def main(argv):
  client_lr_range = [10 ** x for x in range(-4, 0, 1)]
  server_lr_range = [10 ** (x / 10) for x in range(-30, 10, 5)]
  grid = utils_impl.iter_grid({
      'client_learning_rate': client_lr_range,
      'server_learning_rate': server_lr_range})

  py_exec = 'python'
  executable_file = 'tflib/train.py'
  flagfile = '--flagfile {}'.format(FLAGS.flag_file)
  executable = ' '.join(py_exec, executable_file, flagfile)

  utils_impl.launch_experiment(
      executable,
      grid_iter=grid,
      max_workers=FLAGS.max_workers)
  

if __name__ == '__main__':
  app.run(main)
