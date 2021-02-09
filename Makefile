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
	rm -rf torchlib/__pycache__ __pycache__ Node/__pycache__

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
	cd data/server_simulation && python distribute_data.py -s && python calc_class_distribution.py && cd ../..

minimal_server_folders: symbolic_server_folders
	cd data/server_simulation && python delete_all_but_n.py 4 && python calc_class_distribution.py && cd ../..
small_server_folders: symbolic_server_folders
	cd data/server_simulation && python delete_all_but_n.py 10 && python calc_class_distribution.py && cd ../..

# automated script to create image segmentation dataset (jpg) from Medical Decathlon Segmentation dataset 
# possibly easy to adapt for other medical datasets 
create_processed_jpg_from_MSD: 
	@echo Converting raw MSD dataset to jpegs
	python data/MSD/msd_to_jpg.py --data data/MSD/Task03_Liver --res 256 --res_z 64 --crop_height 64 --num_samples 281
	@echo Finished conversion from raw MSD data to jpegs

# Training
federated_secure:
	@echo Training on VirtualWorkers with SecAgg
	python train.py --config configs/torch/pneumonia-resnet-pretrained.ini --train_federated --data_dir data/server_simulation
	@echo Finished Training on VirtualWorkers with SecAgg

# Segmentation 
federated_secure_segmentation:
	@echo Training a Seg-Net on VirtualWorkers with SecAgg
	python train.py --config configs/torch/segmentation.ini --train_federated --cuda --data_dir data/MSD/Task03_Liver
	@echo Finished Training on VirtualWorkers with SecAgg

make federated_DP_secure:
	@echo Training on VirtualWorkers with SecAgg and DP
	python train.py --config configs/torch/pneumonia-resnet-pretrained-DP.ini --train_federated --data_dir data/server_simulation
	@echo Finished Training on VirtualWorkers with SecAgg and DP

# Segmenation 
make federated_DP_secure_segmentation:
	@echo Training a Seg-Net on VirtualWorkers with SecAgg and DP
	python train.py --config configs/torch/segmentation-DP.ini --train_federated --cuda --data_dir data/MSD/Task03_Liver
	@echo Finished Training on VirtualWorkers with SecAgg and DP

federated_insecure:
	@echo Training on VirtualWorkers without SecAgg
	python train.py --config configs/torch/pneumonia-resnet-pretrained.ini --train_federated --data_dir data/server_simulation --unencrypted_aggregation
	@echo Finished Training on VirtualWorkers without SecAgg

# Segmentation 
federated_insecure_segmentation: 
	@echo Training a Seg-Net on VirtualWorkers without SecAgg
	python train.py --config configs/torch/segmentation.ini --train_federated --cuda --unencrypted_aggregation --data_dir data/MSD/Task03_Liver
	@echo Finished Training on VirtualWorkers without SecAgg

federated_gridnode_secure:
	python train.py --config configs/torch/pneumonia-resnet-pretrained.ini --train_federated --websockets --data_dir data/server_simulation

federated_gridnode_insecure:
	python train.py --config configs/torch/pneumonia-resnet-pretrained.ini --train_federated --data_dir data/server_simulation --websockets --unencrypted_aggregation

local:
	@echo Training Locally
	python train.py --config configs/torch/pneumonia-resnet-pretrained.ini --data_dir data/train/ --cuda
	@echo Finished Training Locally

# Segmentation 
local_segmentation:
	@echo Segmentation Training Locally
	python train.py --config configs/torch/segmentation.ini --visdom --cuda --verbose --data_dir data/MSD/Task03_Liver --dump_gradients_every 500
	@echo Finished Training Locally

# Gridnode ensemble shortcut
gridnode:
	python torchlib/run_websocket_server.py --data_dir data/server_simulation --config configs/torch/pneumonia-resnet-pretrained.ini

# Inference
data_owner:
	python -m Node --id data_owner --port 8770 --data_dir .inference --config configs/torch/pneumonia-resnet-pretrained.ini --mean_std_file data/server_simulation/worker1/mean_std.pt

crypto_provider:
	python -m Node --id crypto_provider --port 8780

model_owner:
	python -m Node --id model_owner --port 8771

inference_setup: 
	make data_owner & make crypto_provider & make model_owner

encrypted_inference_local:
	@echo Local encrypted inference
	python inference.py --data_dir .inference --model_weights .pretrained_weights/local_873.pt --encrypted_inference

encrypted_inference_ws:
	@echo Websocket encrypted inference
	python inference.py --data_dir .inference --model_weights .pretrained_weights/local_873.pt --encrypted_inference --websockets_config configs/websetting/config_inference.csv

encrypted_inference_http:
	@echo HTTP encrypted inference
	python inference.py --data_dir .inference --model_weights .pretrained_weights/local_873.pt --encrypted_inference --websockets_config configs/websetting/config_inference.csv --http_protocol

unencrypted_inference_ws:
	@echo Websocket encrypted inference
	python inference.py --data_dir .inference --model_weights .pretrained_weights/local_873.pt --websockets_config configs/websetting/config_inference.csv

unencrypted_inference_http:
	@echo HTTP encrypted inference
	python inference.py --data_dir .inference --model_weights .pretrained_weights/local_873.pt --websockets_config configs/websetting/config_inference.csv --http_protocol	

# Hyperparam-search 
private_seg_search: 
	python torchlib/find_config_seg.py --trial_name "MoNet_pretrained_local_1" --data_dir data/MSD/Task03_Liver 