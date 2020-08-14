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

install:
	conda env create -f environment_torch.yml

minimal_server_folders: symbolic_server_folders
	cd data/server_simulation && python delete_all_but_n.py 16 && python calc_class_distribution.py && cd ../..

crypten_dataset:
	@echo FAILS
	python data/create_crypten_data.py

federated_secure:
	@echo Training on VirtualWorkers with SecAgg
	python train.py --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --data_dir data/server_simulation

federated_insecure:
	@echo Training on VirtualWorkers without SecAgg
	python train.py --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --data_dir data/server_simulation --unencrypted_aggregation

local_secure:
	@echo Training Locally with SecAgg
	python train.py --config configs/torch/pneumonia-resnet-pretrained-fast.ini --data_dir data/server_simulation/worker1

local_insecure:
	@echo Training Locally without SecAgg
	python train.py --config configs/torch/pneumonia-resnet-pretrained-fast.ini --data_dir data/server_simulation/worker1 --unencrypted_aggregation

local_secure_cuda:
	@echo Training Locally with SecAgg on CUDA
	python train.py --config configs/torch/pneumonia-resnet-pretrained-fast.ini --data_dir data/server_simulation/worker1 --cuda

local_insecure_cuda:
	@echo Training Locally without SecAgg on CUDA
	python train.py --config configs/torch/pneumonia-resnet-pretrained-fast.ini --data_dir data/server_simulation/worker1 --unencrypted_aggregation --cuda

assert_cuda_fail:
	@echo Training Federated with CUDA -> Designed to fail. Does not exit with code 1.
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --cuda

train_all: federated_secure federated_insecure local_insecure local_secure_cuda local_insecure_cuda

visdom_train:
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --visdom

crypten_benchmark:
	@echo WARNING: For this to work, make sure there is a .pretrained_weights directory in the repository root and that it contains a file called crypten_weights.pt.
	@echo WARNING: This will probably strain your computer A LOT! 
	python torchlib/crypten_inference.py --model_weights .pretrained_weights/crypten_weights.pt --max_num_samples 2 --batch_size 1