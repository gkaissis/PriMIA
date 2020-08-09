clean_python:
	rm -rf .mypy_cache
	rm -rf torchlib/__pycache__

clean_weights:
	rm -rf model_weights

clean_server_folders:
	cd data/server_simulation && rm -rf all_samples/ validation/ worker1 worker2 worker3 && cd ../..

clean_all: clean_python clean_weights clean_server_folders

server_folders:
	cd data/server_simulation && python distribute_data.py && cd ../..

symbolic_server_folders:
	cd data/server_simulation && python distribute_data.py -s && cd ../..

dependencies:
	conda env create -f environment_torch.yml

pylint:
	pylint torchlib

minimal_server_folders: symbolic_server_folders
	cd data/server_simulation && python delete_all_but_one.py && cd ../..

fast_virtualtrain:
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --no_visdom --no_cuda

secure_aggregation:
	echo This will probably fail!
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --no_visdom --no_cuda --secure_aggregation
