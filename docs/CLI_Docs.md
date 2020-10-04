# Library Documentation

## Video walkthrough
A 50-minute video walkthrough for PriMIA can be found [here](https://youtu.be/LRF2l21n6F0)

## Documentation for the individual components
This section describes what the individual components of PriMIA do.

- `configs`
This folder contains the configuration files. `torch` contains the `config.ini` files which are used for configuring the training process. `websetting` contains files for usage with WebSocket/HTTP training over the network. Documentation for these files can be found [here](ConfigDocs.md)

- `data`
This folder contains the [dataset](Dataset_Description.md) used in our publication and case study. After running `make <symbolic/minimal>_server folders`, a `server_simulation` folder appears here which holds the data for the VirtualWorker nodes. The `train` and `test` folders hold the original dataset. The two `.csv` files contain metadata and labels.

- `docs`
This folder contains the documentation source and is not needed by the end user.

- `figure_scripts`
This folder contains scripts for re-creating the figures for our publication. It is not needed by the end user.

- `model_weights`
This folder does not always exist. It will be generated after a training run and stores model weights and a `.csv` file with the training metrics.

- `Node`
This folder contains the code for setting up a PyGrid node. Documentation for how to use this can be found [here](HowTo.md)

- `site`
This folder contains the HTML and CSS for the documentation. It should not be changed by the end user

- `syft`
This folder contains the PySyft source code. It should not be altered unless you really know what you are doing.

- `torchlib`
This folder contains all the utility modules (the _engine room_) for PriMIA. 
    - `dataloader.py` contains the `CombinedLoader` class which is the main dataloader for PriMIA and loads image and medical image files. It also contains various utility functions required for PriMIA's operation and ways to load both the supplied paediatric pneumonia dataset and the MNIST dataset which can be used for experimentation.
    - `dicomtools.py` contains the logic for loading DICOM files. Documentation for the module can be found in the module's source.
    - `find_config.py` contains the logic to run hyperparameter optimisation with Optuna. Its documentation can be found [here](HowTo.md)
    - `models.py` contains the models available in PriMIA, a simple CNN (called `simpleconv`), a VGG-style CNN and the ResNet18 used in our publication
    - `run_websocket_server.py` is a utility script to quickly run a set of WebSocket/HTTP servers to train models. It is not required for operation since servers can be run directly from the `Node` module.
    - `utils.py` contains the entire logic for federated learning. If you are interested in the internals of PriMIA, this is where to look.

- `DATASET_LICENSE`
This is the license for the dataset used, which is different from the source code license. Please read the license and use PriMIA and the data _only_ in accordance to its terms.

- `doc_requirements.txt`
This is the requirements file to build the documentation. It is normally not needed by the end user.

- `environment_torch.yml`
This is the preferred way to install dependencies. See [here](HowTo.md) for more information on setting up dependencies and an environment to run PriMIA.

- `inference.py`
This is the entrypoint to perform (encrypted inference) with PriMIA and implements the core logic. If you are interested in the internals of encrypted inference, this is the file you are looking for.

- `LICENSE`
This is the license for the source code, which is different from the dataset license. Please read the license and use PriMIA and the data _only_ in accordance to its terms.

- `Makefile`
This file contains a large amount of pre-made _recipes_ which can `make` (pun intended) your life much easier. See [here](HowTo.md) for how to use these recipes.

- `mkdocs.yml`
This is a configuration file for the documentation. It is normally not needed by the end user

- `README.md`
This contains the GitHub readme. It is not needed by the end user.

- `requirements.txt`
This is the other (and not preferred) way to install dependencies. See [here](HowTo.md) for more information on setting up dependencies and an environment to run PriMIA.

- `test.py`
This is the entrypoint to test a trained algorithm on data which is already set up in named folders identical to the ones in our dataset. It is a convenience script to obtain model metrics and does not represent the main use-case for PriMIA. 

- `train.py`
This is the entrypoint to federated or local model training and contains all the logic of the training loop. If you are interested in how training in PriMIA is carried out, this is the script to look at.