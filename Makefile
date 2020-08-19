#Setup
install:
	conda env create -f environment_torch.yml

update:
	conda env update -f environment_torch.yml

doc_install:
	pip install -rU doc_requirements.txt

#Cleanup
clean_python:
	rm -rf .mypy_cache
	rm -rf torchlib/__pycache__

clean_weights:
	rm -rf model_weights

clean_server_folders:
	cd data/server_simulation && rm -rf all_samples/ validation/ worker1 worker2 worker3 && cd ../..

clean_mnist:
	cd data/ && rm -rf LabelMNIST && cd ../..

clean_crypten:
	cd data && rm -f testdata.pt testlabels.pt && cd ../..

clean_all: clean_python clean_weights clean_server_folders clean_crypten clean_mnist

#Create Datasets
server_folders:
	cd data/server_simulation && python distribute_data.py && cd ../..

symbolic_server_folders:
	cd data/server_simulation && python distribute_data.py -s && cd ../..

minimal_server_folders: symbolic_server_folders
	cd data/server_simulation && python delete_all_but_n.py 16 && python calc_class_distribution.py && cd ../..

#CrypTen Benchmark
crypten_dataset:
	@echo Creating CrypTen dataset from the test set
	python data/create_crypten_data.py

crypten_benchmark:
	@echo WARNING: For this to work, make sure there is a .pretrained_weights directory in the repository root and that it contains a file called crypten_weights.pt.
	@echo WARNING: This will probably strain your computer A LOT! 
	python torchlib/crypten_inference.py --model_weights .pretrained_weights/crypten_weights.pt --max_num_samples 2 --batch_size 1

################ TESTS ######################
federated_secure:
	@echo Training on VirtualWorkers with SecAgg
	python train.py --config configs/test_configs/weighted_classes.ini --train_federated --data_dir data/server_simulation
	@echo Finished Training on VirtualWorkers with SecAgg

federated_gridnode:
	python train.py --config configs/torch/pneumonia-resnet-pretrained.ini --train_federated --data_dir data/server_simulation --websockets

federated_insecure:
	@echo Training on VirtualWorkers without SecAgg
	python train.py --config configs/test_configs/weighted_classes.ini --train_federated --data_dir data/server_simulation --unencrypted_aggregation
	@echo Finished Training on VirtualWorkers without SecAgg

local:
	@echo Training Locally
	python train.py --config configs/test_configs/non_weighted_classes.ini --data_dir data/server_simulation/worker1 
	@echo Finished Training Locally

local_cuda:
	@echo Training Locally with SecAgg on CUDA
	python train.py --config configs/test_configs/non_weighted_classes.ini --data_dir data/server_simulation/worker1 --cuda

assert_cuda_fail:
	@echo Training Federated with CUDA. Designed to fail. Does not exit with code 1.
	python train.py --config configs/test_configs/weighted_classes.ini --data_dir data/server_simulation/worker1 --train_federated --cuda

train_all: federated_secure federated_insecure local local_cuda assert_cuda_fail
	@echo All checks successful

###### VISDOM
federated_secure_visdom:
	@echo Training on VirtualWorkers with SecAgg
	python train.py --config configs/test_configs/visdom.ini --train_federated --data_dir data/server_simulation --visdom
	@echo Finished Training on VirtualWorkers with SecAgg

###### VISDOM
mixup_ablation:
	python train.py --config configs/test_configs/mixup_ablation.ini --data_dir data/server_simulation/worker1 --visdom --cuda
	
gridnode:
	python torchlib/run_websocket_server.py --data_dir data/server_simulation --config configs/torch/pneumonia-resnet-pretrained.ini