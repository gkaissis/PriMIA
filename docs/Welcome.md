# Start here

## What do you want to do?

- Create a conda environment and install all dependencies to run our code:
    - Only tested on Ubuntu 20.04 LTS
    - Run `make dependencies`

- Simulate training with VirtualWorkers using our data:
    - Run `make symbolic_server_folders`
    - Run `make fast_virtualtrain`
    - The model weights will be saved under `model_weights`

- Simulate training with WebSocketWorkers using our data:
    - Run run_websocket_server.py ... (finds data automatically)
    - #TODO add instructions for GridNode

- Run training on VirtualWorkers using your own data:
    - #TODO add instructions (...) populate folders etc.

- Run training on real servers using WebSockets and your own data:
    - run websocket server on each cloud instance (finds data automatically)
    - #TODO add instructions

- Create a minimal training and validation set of 1 image each to debug:
    - Run `make minimal_server_folders`

- Clean up everything and restore the environment to its default state:
    - Run `make clean_all`

- Run a benchmark with CrypTen:
    WARNING: This is EXTREMELY processor and RAM intense and can result in your machine running out of RAM or locking up.
    - Make sure there is a `.pretrained_weights` directory in the project root containing the model you want to encrypt
    - Name the model you would like to use `crypten_weights.pt``
    - Run `make crypten_dataset`
    - Run `make crypten_benchmark`
    

- Run MNIST/Run VGG etc. etc. 
    - args ...

- Monitor your training with visdom

- Run a hyperparam optimisation trial (locally)
