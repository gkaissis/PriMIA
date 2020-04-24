# Private Paediatric Pneumonia Prediction (4P)

## Dataset 
Please check [this file](docs/Dataset_Description.md) for information on how to use this dataset.

## Allocating data to clients for federated learning
The [client_data.py](common/client_data.py) script can be used to allocate the dataset to specific clients. All federated training code should rely on this script to generate data for experiments, and should make note of any changes to default parameters for reproducibility.

You can use the script by running `python client_data.py` at the command line. This will populate the `data/train_clients/` and `data/test_clients/` directories with client-allocated data. See the script for configuration options (or use `python client_data.py -h`).

___

The dataset in this repository is being re-used under the license terms from [the CoronaHack Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset) and modified as indicated in `Dataset_Description.md`.

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
