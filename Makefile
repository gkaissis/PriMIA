clean:
	rm -rf .mypy_cache

clean_training: clean
	rm -rf model_weights

dependencies:
	conda env create -f environment_torch.yml

pylint:
	pylint torchlib

fast_train:
	python train.py --dataset pneumonia --config configs/torch/pneumonia-resnet-pretrained-fast.ini --train_federated --no_visdom --no_cuda
