import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import syft as sy
import torch

torch.set_num_threads(1)

hook = sy.TorchHook(torch)
client = sy.workers.node_client.NodeClient(hook, "http://127.0.0.1:8777", id="alice")
print(client)
grid = sy.PrivateGridNetwork(client,)
print(grid)
data = grid.search("#mnist", "#data")
target = grid.search("#mnist", "#target")
dataset = [sy.BaseDataset(data[worker][0], target[worker][0]) for worker in data.keys()]
fed_datasets = sy.FederatedDataset(dataset)
train_loader = sy.FederatedDataLoader(fed_datasets)
print(train_loader)
for i, d in enumerate(train_loader):
    print(i)
"""
data = grid.search("pneumonia_data")
print(data)
inputs = sy.local_worker.request_search("pneumonia", location=client)
print(inputs)
"""
# labels = client.search("labels")
# print(labels)
print("done")
