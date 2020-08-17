# How-To Guides

Here you can find a collection of recipes for various tasks and settings you might want to try out.

## What do you want to do?

### 1. Dependencies and Environment
- Create a conda environment and install all dependencies to run our code and reproduce our results:
    - Create a new conda environment with python 3.7.1. Other pythons are not supported!
    - Run `make install` to recreate the environment
    - Disclaimer: We have only tested our code on Ubuntu 18.04 and 20.04. We cannot guarantee compatibility with any other operating system.
    
- Clean up everything and restore the environment to its default state:
    - Run `make clean_all`
    
    
### 2. Training using VirtualWorkers

- Simulate training with VirtualWorkers using the Paediatric Pneumonia dataset:
    - Run `make symbolic_server_folders` to randomly split the Paediatric Pneumonia Dataset into three worker folders and a validation set. You can also run `make minimal_server_folders` if you are in a real hurry, but results won't be nearly as good.
    - Run `make federated_secure` to train a model with encrypted federated learning
    - The model weights and a `.csv` file with metadata will be saved under `model_weights`


- Run training on VirtualWorkers using your own data:
    - TODO: add instructions (...) populate folders etc.
    - If you want to use more or fewer workers

### 3. Training using GridNodes

- Simulate training with GridNodes using our data:
    - Run run_websocket_server.py ... (finds data automatically)
    - TODO: add instructions for GridNode
    - If you want to use more or fewer workers

- Run training on real servers using GridNodes and your own data:
    - run websocket server on each cloud instance (finds data automatically)
    - TODO: add instructions
    - If you want to use more or fewer workers

### 4. CrypTen

- Use the CrypTen library to perform an inference benchmark on the Paediatric Pneumonia Test Set:
    WARNING: This is EXTREMELY processor and RAM intense and can result in your machine running out of RAM or locking up.
    - Make sure there is a `.pretrained_weights` directory in the project root containing the model you want to encrypt
    - Name the model you would like to use `crypten_weights.pt`
    - Run `make crypten_dataset`. This will create `testdata.pt` and `testlabels.pt` under `data/`. These correspond to the test set images and labels.
    - Run `make crypten_benchmark`. This will run a benchmark on two images from the test set and should take about 200 seconds on a powerful machine. It might take **much** longer depending on your CPU and RAM.

### 5. Miscellaneous

- If you want to adapt PriMIA to another use-case altogether, go [here](HoWToMod.md)

- Monitor your training with Visdom
    - From the command line, run `visdom` to start a visdom server. It will be located on `localhost:xxxx`. Navigate to this page with your browser.
    - For an example, run `make federated_secure_visdom` which will launch a slightly longer training which you can monitor with Visdom! 
    - For more information on Visdom, see [here](https://github.com/facebookresearch/visdom). Scroll down on this page to see configuration options, e.g. port selection or authentication for Visdom and many tutorials!

- Run MNIST/Run VGG etc. etc. 
    - args ...


- Run a hyperparam optimisation trial (locally)

- Run a hyperparameter optimisation trial for the whole federation
