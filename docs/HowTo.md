# How-To Guides

## What do you want to do?

### 1. Dependencies and Environment
- Create a conda environment and install all dependencies to run our code:
    - Only tested on Ubuntu 20.04 LTS
    - Run `make dependencies`

- Clean up everything and restore the environment to its default state:
    - Run `make clean_all`

### 2. Training using VirtualWorkers

- Train with VirtualWorkers using the Paediatric Pneumonia dataset:
    - Run `make symbolic_server_folders`
    - Run `make fast_virtualtrain`
    - The model weights will be saved under `model_weights`

- Run training on VirtualWorkers using your own data:
    - TODO: add instructions (...) populate folders etc.

### 3. Training using GridNodes

- Simulate training with GridNodes using our data:
    - Run run_websocket_server.py ... (finds data automatically)
    - TODO: add instructions for GridNode

- Run training on real servers using GridNodes and your own data:
    - run websocket server on each cloud instance (finds data automatically)
    - TODO: add instructions

### 4. CrypTen

- Use the CrypTen library to perform an inference benchmark on the Paediatric Pneumonia Test Set:
    WARNING: This is EXTREMELY processor and RAM intense and can result in your machine running out of RAM or locking up.
    - Make sure there is a `.pretrained_weights` directory in the project root containing the model you want to encrypt
    - Name the model you would like to use `crypten_weights.pt`
    - Run `make crypten_dataset`. This will create `testdata.pt` and `testlabels.pt` under `data/`. These correspond to the test set images and labels.
    - Run `make crypten_benchmark`. This will run a benchmark on two images from the test set and should take about 200 seconds on a powerful machine. It might take **much** longer depending on your CPU and RAM.

### 5. Miscellaneous

- Run MNIST/Run VGG etc. etc. 
    - args ...

- Monitor your training with visdom

- Run a hyperparam optimisation trial (locally)
