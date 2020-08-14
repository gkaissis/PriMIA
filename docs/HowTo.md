# How-To Guides

Here you can find a collection of recipes for various tasks and settings you might want to try out.

## What do you want to do?

- Create a conda environment and install all dependencies to run our code and reproduce our results:
    - Create a new conda environment with python 3.7.1. Other pythons are not supported!
    - Run `make install` to recreate the environment
    - Disclaimer: We have only tested our code on Ubuntu 18.04 and 20.04. We cannot guarantee compatibility with any other operating system.

- Simulate training with VirtualWorkers using our data:
    - Run `make symbolic_server_folders` to randomly split the Paediatric Pneumonia Dataset into three worker folders and a validation set. You can also run `make minimal_server_folders` if you are in a real hurry, but results won't be nearly as good.
    - Run `make federated_secure` to train a model with encrypted federated learning
    - The model weights and a `.csv` file with metadata will be saved under `model_weights`

- Simulate training with WebSocketWorkers using our data:
    - Run run_websocket_server.py ... (finds data automatically)
    - TODO: add instructions for GridNode

- Run training on VirtualWorkers using your own data:
    - TODO: add instructions (...) populate folders etc.

- Run training on real servers using WebSockets and your own data:
    - run websocket server on each cloud instance (finds data automatically)
    - TODO: add instructions

- Create a minimal training and validation set of 1 image each to debug:
    - Run `make minimal_server_folders`

- Clean up everything and restore the environment to its default state:
    - Run `make clean_all`

- Run a benchmark with CrypTen:
    WARNING: This is EXTREMELY processor and RAM intense and can result in your machine running out of RAM or locking up.
    - Make sure there is a `.pretrained_weights` directory in the project root containing the model you want to encrypt
    - Name the model you would like to use `crypten_weights.pt`
    - Run `make crypten_dataset`
    - Run `make crypten_benchmark`
    

- Run MNIST/Run VGG etc. etc. 
    - args ...

- Monitor your training with visdom

- Run a hyperparam optimisation trial (locally)

- Run a hyperparameter optimisation trial for the whole federation
