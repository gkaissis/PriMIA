import syft as sy
import torch
import os

hook = sy.TorchHook(torch)
client = sy.workers.node_client.NodeClient(hook, "http://127.0.0.1:8777", id="alice")
print(client)
inputs = sy.local_worker.request_search("pneumonia", location=client)
print(inputs)
# labels = client.search("labels")
# print(labels)
print("done")
