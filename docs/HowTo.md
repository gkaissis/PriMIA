# How-To Guides

Here you can find a collection of recipes for various tasks and settings you might want to try out.

## What do you want to do?

### 1. Dependencies and Environment
- Create a `conda` environment and install all dependencies to run our code and reproduce our results:
    - Create a new `conda` environment with python 3.7.1. Other pythons are not supported!
    - Run `make install` to recreate the environment
    - `make update` can be used to update the conda environment from the `.yml` file.
    - The environment will be called `torch4p`
    - Disclaimer: We have only tested our code on Ubuntu 18.04 and 20.04. We cannot guarantee compatibility with any other operating system.
    - macOS users: PriMIA _should_ work on macOS, however only using `pip` as described below

- Use `pip` to set up an environment:
    - We discourage this method and do not support it, so proceed at your own risk
    - Make a virtual environment using the tool of your choice and using python 3.7.1. Other pythons are not supported!
    - Run `pip install -r requirements.txt` to install the basic requirements
    - After these have installed, run `pip install -U syft==0.2.9` to install PySyft
    - `pip` will complain (quite loudly and in red as of September 2020) about inconsistencies with `torchdp`. These can be ignored
    
- Clean up everything and restore the environment to its default state:
    - Run `make clean_all`. Please be **careful** when running this, as it **DELETES** folders, including the `model_weights` folder which perhaps stores your training weights! For a detailed list of what this command destroys, check the `Makefile` under `Cleanup`.

### 2. Work with the Paediatric Pneumonia Dataset
- Distribute the data to individual worker folders:
    - Run `make server_folders`. This will actually copy the files to `data/server_simulation/worker<i>`. It can be useful if you want to use the data on remote machines instead of locally
    - Run `make symbolic_server_folders` if you intend to work locally only. This creates symbolic links, saves a lot of space and is faster
    - Run `make minimal_server_folders` to create a minimal dataset of 4 images (this can be modified in the `Makefile`) per worker (times the `repetitions_dataset` parameter from the `config.ini` file) for quickly trying out something after e.g. making changes to the code.

### 3. Work with the MNIST dataset
This does not happen from the Makefile but rather is passed as a flag to the training scripts.
    
    
### 4. Training using VirtualWorkers

- Train using VirtualWorkers using the Paediatric Pneumonia dataset (quick way):
    - Run `make symbolic_server_folders` to randomly split the Paediatric Pneumonia Dataset into three worker folders and a validation set. You can also run `make minimal_server_folders` if you are in a real hurry, but results will be predictably poor.
    - Run `make federated_secure` to train a model using federated learning with secure aggregation. `make federated_insecure` can be used to suppress secure aggregation
    - The model weights and a `.csv` file with metadata will be saved under `model_weights`

- Train using VirtualWorkers using the Paediatric Pneumonia dataset (slow way):
    - Make whichever modifications you need to the configuration file (documentation [here](ConfigDocs.md))
    - Run `python train.py --config <path/to/your/config.ini> --train_federated --data_dir data/server_simulation` 
    - Pass the `--unencrypted_aggregation` flag to suppress secure aggregation.

- Train on your own paediatric pneumonia data:
    - pass `--data_dir <path/to/your/data>` to the CLI    

### 5. Training using PyGrid Nodes

- Set up PyGrid Nodes on your local machine and run training with them
    - Run `make gridnode`. This assumes you are using the Paediatric Pneumonia Dataset and the pre-made `pneumonia-resnet-pretrained.ini` file and is a convenience function. Make adjustments to the `Makefile` or directly run the following if you require more flexibility: `python torchlib/run_websocket_server.py --data_dir data/server_simulation --config <path/to/your/config.ini>`
    - Run `make federated_gridnode_secure` to train on the GridNodes with secure aggregation or `make federated_gridnode_insecure` to eschew secure aggregation

- Set up a PyGrid Node on a local or remote server for federated training
    - Run `python -m Node --id <desired id> --port <desired port> --data_dir <path/to/data> --config <path/to/config.ini>`. The configuration file must be identical on all remote servers and the central server. The name of the node, the IP address and the port must be changed in `websetting/config.csv`
    - When training, the training coordinator/ central server must pass the `--websockets` flag to train.py which will read the settings from `websetting/config.csv` and configure the connections automatically.

Sidenote: The number of workers can be changed by omitting workers from the configuration `csv` file.

### 6. Run training locally using GPUs
Run `make local`. Alternatively, run `python train.py --config <path/to/your/config.ini> --data_dir <path/to/data> --cuda`.

Note that macOS has no CUDA support. In this case, `--cuda` will do nothing. It can also be omitted if training on CPU is desired.

### 7. Miscellaneous Training

- If you want to adapt PriMIA to another use-case altogether, go [here](HowToMod.md)

- Monitor your training with Visdom
    - From the command line, run `visdom` to start a visdom server. It will be located on `localhost:xxxx`. Navigate to this page with your browser.
    - Add the `--visdom` flag to train.py 
    - For more information on Visdom, see [here](https://github.com/facebookresearch/visdom). Scroll down to see configuration options, e.g. port selection or authentication for Visdom and many tutorials!

- Use MNIST, VGG16 etc.
    - These are handled using command line arguments. `--data dir mnist` will use MNIST from `torchvision`. The model can be switched in the configuration file. Check [this page](ConfigDocs.md) for details.

- Run a hyperparameter optimisation trial
    - Run `python torchlib/find_config.py`. This assumes the system is set up for training (as described above). PriMIA uses [Optuna](https://optuna.readthedocs.io/en/stable/#). The system defaults to local training. If VirtualWorkers are required, pass the `--federated` flag. If PyGrid nodes are running, you can pass `--websockets` (which will be passed on to `train.py`). A database file can be specified here, otherwise a default SQLite file will be used.
    - Results can be visualised running the script with the `--visualize` flag, which will read the database file and open an Optuna server to show the results
    - The results of the hyperparameter run will be located inside `model_weights`. If running many trials, make sure you have enough space available since this folder will become very large.

- Differential privacy
    - PriMIA includes bindings for the `torchdp` library (now called [Opacus](https://github.com/pytorch/opacus)). Differential privacy is only implemented for simple models at the moment and is in an experimental stage.

### 8. Inference
- Run inference with VirtualWorkers
    - Put data to classify in a directory
    - Have a trained model ready (in `.pt` format)
    - Run `python inference.py --data_dir <path/to/data> --model_weights <path/to/model> --encrypted_inference`. The `Makefile` also provides some premade recipes which need to be adapted to your data and models.
    - CAUTION: Encrypted inference is **extremely** resource intensive and can cause your computer to become unresponsive or the process to be killed by your operating system. Omit the `--encrypted_inference` flag to perform regular remote inference
    - On compatible systems, inference can be accelerated with `--cuda`
    - Do not confuse inference with using the `test.py` file. This is a convenience script that will only work with the pneumonia dataset used in our publication.

- Run inference over the network
    - If the `--websockets_config` flag is passed alongside the path to a `configuration.ini` file (a template can be found in `configs/websetting`), inference will be performed over the network. The ports and IP addresses must match the ports and IP addresses of your machines.
    - This requires PyGrid nodes to be set up as a data owner, a model owner and a crypto provider. The `Makefile` provides some templates for this. If you want to simulate this process locally, you can run `make inference_setup` in one terminal, then run `inference.py` in a different terminal.
    - For best results, you should pass a `mean_std_file` for inference, which contains the mean and standard deviation of the training data which is used for re-scaling the incoming data and is generated automatically during the training process. If this is omitted, sensible defaults are used. 
    - Encrypted inference over the network is non-trivial, since the underlying WebSocket implementation has an issue where kernel TCP buffers can overflow (see issue [here](https://github.com/websocket-client/websocket-client/issues/314)). Provided you _really_ know what you are doing, you can tune the buffers using [this guide](https://www.cyberciti.biz/faq/linux-tcp-tuning/). If you experience lag, delays or performance degradation, this is likely a problem with either your network settings or hardware. PriMIA does not interact directly with any networking layer.
    - Alternatively, we provide the option of using HTTP exclusively for inference. This is slightly slower (as HTTP is not full duplex) and requires more I/O as it uses base-64 encoding. It is rock-stable though and can be enabled with the `--http_protocol` flag.
    - TLS is handled by PyGrid, not PriMIA. If you have certificates and want to use WSS or HTTPS, these need to be loaded onto the Nodes manually. Furthermore, the Nodes produce a warning related to the secret key, which should not be left at its default setting for security purposes, but passed as an environment variable. More info can be found [here](https://github.com/OpenMined/PyGrid).
    - Encrypted inference is very resource and I/O intensive.  

