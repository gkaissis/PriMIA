# Configuration File Documentation

This file contains the documentation for the configuration files, which can be found inside the `configs` directory. 
The directory has two subdirectories, `torch`, which contains the training-related configuration files and `websetting` which contains WebSocket/HTTP related configuration files. 

## `torch` subdirectory

The training-related configuration files can be found or placed here. A series of pre-made configuration files with self-explanatory names are available. We suggest adapting `pneumonia-resnet-pretrained.ini` to your purpose. This file will be required by PriMIA whenever a `--config` flag is required in the CLI. The structure of the file can be found below. Documentation for non self-explanatory items is included as a comment. 
 
```python
[config]
batch_size = 4
train_resolution = 224
inference_resolution = 512
test_batch_size = 1
test_interval = 1 # How often to test the model during training. 
                  # 1 means "every epoch"
validation_split = 10 # in %
epochs = 1 # How many epochs. If left empty defaults to 40
lr = 1e-4 # Initial learning rate
end_lr = 1e-5 # Final learning rate after decay
restarts = 0 # Warm restarting of the learning rate. 0 means no restarts
beta1 = 0.5 # Adam optimiser parameter
beta2 = 0.99 # Adam optimiser parameter
weight_decay = 5e-4 # =L2 regularisation
;momentum = 0.5 # SGD momentum term
deterministic = yes # use random seed and torch deterministic options
                    # WARNING: this is incompatible with certain 
                    # settings and will give an appropriate warning
seed = 1 # random seed
log_interval = 10 # used in conjunction with visdom. 
                  # How often to log results to the visdom server
optimizer = Adam # "SGD" or "Adam"
differentially_private = no # use differential privacy. This feature is 
                            # experimental and not supported on complex
                            # network architectures
model = resnet-18 # "simpleconv", "vgg16" or "resnet-18"
pretrained = yes # Use ImageNet weights
weight_classes = yes # Use class-weighted gradient descent
pooling_type = max # "max" or "avg"

[augmentation]
rotation = 30
translate = 0.0
scale = 0.15
shear = 10
mixup = yes # use MixUp augmentation
mixup_lambda = 0.5 # mixing parameter. Leaving it empty results 
                   # in stochastic MixUp where the parameter
                   # gets sampled from a random uniform 
                   # distribution at every application
mixup_prob = 0.9 # MixUp probability

[albumentations] # Settings passed to albumentations
clahe = yes # contrast-limited adaptive histogram equalisation
overall_prob = 0.75 # overall probability of applying albumentations
individual_probs = 0.2 # probabilities of the individual augmentations
noise_std = 0.05 # standard deviation of the Gaussian noise
noise_prob = 0.5 # probability of adding noise
randomgamma = yes
randombrightness = yes
blur = yes
elastic = yes
optical_distortion = yes
grid_distortion = yes
grid_shuffle = no
hsv = no # HSV space augmentations
invert = no
cutout = no
shadow = no
fog = yes
sun_flare = no
solarize = no
equalize = no
grid_dropout = no

[federated]
sync_every_n_batch = 2 # Synchronisation parameter. 
                       # High values make training take longer
                       # but usually result in better performance
wait_interval = 0.1 # not implemented
keep_optim_dict = no # not implemented
repetitions_dataset = 1 # how many times to augment/repeat the 
                        # dataset on the nodes. 
                        # Will adapt based on number of epochs
weighted_averaging = yes # class-weighted federated averaging

[system]
num_threads = 16 # passed to PyTorch
```

## `websetting` subdirectory
This contains two files, `config.csv` which should _not_ be renamed but can be changed and `config_inference.csv` which can be renamed and is required whenever PriMIA asks for a `--websockets_config` file.

`config.csv` is used by the utility script `run_websocket_server.py` in the `torchlib` directory. It is not required for normal operation of the library but rather exists as a convenience to quickly set up WebSocket/HTTP-based workers on the local system. It can be altered to change the IP addresses and ports the workers will use from their defaults. The names of the workers can also be altered, but we _strongly_ discourage this unless you understand the consequences, as the names are used for the MNIST training example.

`config_inference.csv` is used for remote inference. The IP addresses of the model owner, the data owner and the crypto provider as well as the ports can be set here. Please note that this is only for establishing a connection to nodes which are already running. If you want to set up nodes, this can be achieved by running the `Node` module with the documentation found [here](HowTo.md)

