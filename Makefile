clean_python:
	rm -rf .mypy_cache
	rm -rf torchlib/__pycache__

clean_weights:
	rm -rf model_weights

clean_server_folders:
	cd data/server_simulation && rm -rf all_samples/ validation/ worker1 worker2 worker3 && cd ../..

clean_crypten:
	cd data && rm -f testdata.pt testlabels.pt && cd ../..

clean_all: clean_python clean_weights clean_server_folders clean_crypten

server_folders:
	cd data/server_simulation && python distribute_data.py && cd ../..

symbolic_server_folders:
	cd data/server_simulation && python distribute_data.py -s && cd ../..

dependencies:
	conda env create -f environment_torch.yml

pylint:
	pylint torchlib

minimal_server_folders: symbolic_server_folders
	cd data/server_simulation && python delete_all_but_n.py 16 && python calc_class_distribution.py && cd ../..

crypten_dataset:
	python data/create_crypten_data.py

fast_virtualtrain:
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated

secure_aggregation:
	@echo This will probably fail!
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated

assert_cuda_fail:
	@echo Designed to fail!
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --cuda

visdom_train:
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --visdom

crypten_benchmark:
	@echo WARNING: For this to work, make sure there is a .pretrained_weights directory in the repository root and that it contains a file called crypten_weights.pt.
	@echo WARNING: This will probably strain your computer A LOT! 
	python torchlib/crypten_inference.py --model_weights .pretrained_weights/crypten_weights.pt --max_num_samples 2 --batch_size 1