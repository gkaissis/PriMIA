import syft as sy
import torch
import os

hook = sy.TorchHook(torch)
client = sy.workers.node_client.NodeClient(hook, "http://127.0.0.1:8777", id="alice")
print(client)
grid = sy.PrivateGridNetwork(client,)
print(grid)
ds = grid.search('mnist')
dss = [ds[0] for ds in ds.values()]
fed_datasets = sy.FederatedDataset(dss)
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
