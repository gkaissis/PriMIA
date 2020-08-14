# Command-line Interface Documentation

- train.py: Main entrypoint for training
    - Train virtually
    - Train on GridNodes (after setting up)
    - Requires config.ini and potentially config.csv settings for the actual IP addresses used in your case
    - Requires DataFolders with data in them (either locally for virtual or remotely for Grid)
    - Will accept cuda as an argument but will always raise error
    - Visdom
    - Dataset choice: MNIST vs. Pneumonia (which we ship with the repo) vs. your own data (which you have to provide)
    - Network choice via config. ini
    - Other parameters via config.ini

- Config.ini documentation

- find_config_federated.py: Random hyperam search for the whole federation (if you are the training coordinator)
    -TODO: hyperparam.ini

- find_config.py: Optuna-based hyperparam search for the local worker (If you are on a local worker)

- inference.py: Entrypoint for performing inference (encrypted or plain text) TODO: remove option for class_report
    - Virtual or Grid
    - Secure or non-secure
    - DataFolder
    - Returns labels/classification_report to stdout

- crypten_inference.py
    - Benchmark for CrypTen on "VirtualWorker equivalent"
    - Needs `make crypten_dataset` 
    - Needs path to some model for performing inference
    - Consider shipping some weights which are default for reproducing results on shipped test data

- run_websocket_server.py: Starts all GridNodes in config.csv -> For simulating GridNodes on localhost
- GridNode/grid_node.py starts local gridnode (must be run on e.g. cloud instance to receive signals from central server) for real world, training coordinator runs train.py with your IP

- create_crypten_data.py: Utility script creates pytorch saved tensor from test set for use with CrypTen script

- server_simulation: Main location for data folders for virtual training/ GridNode simulation TODO: Gets refactored in new branch, rename
    - Scripts calc_distribution -> shows distribution per class in worker folders
    - delete all: cleans out worker folders
    - delet all but n: SelfExp
    - distribute: SelfExp

- websetting config: see above
- torch config: see above

- .pretrained_weights: weights TODO: potentially LFS

- environment_torch.yml: env file for conda TODO: check needed/clean out

- LICENSE: TODO: Different license for data and code

- mkdocs.yml: Internal use

- README: TODO: Farkas and texts

- Makefile: see above and how-to

Removed in release version:
- create_labels.py (-> renamed to test.py, only for internal use since provides class_report)
- eval_expert_labels.py 
- modelsummary.py
- test_dataloader.py
- various CSV files: Deprecated with ImageFolder EXCEPT Labels.csv
- Removed, Adults
- doc_requirements: 

Internal use:
- dataloader.py
- models.py
- utils.py : TODO: needs docstings and tests @gkaissis
- websocket_utils.py TODO: cleanup

