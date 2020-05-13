import torch
import syft as sy
from pandas import read_csv


def start_webserver(id: int, port: str):
    hook = sy.TorchHook(torch)
    server = sy.workers.websocket_server.WebsocketServerWorker(
        id=id, host=None, port=port, hook=hook, verbose=False
    )
    server.start()
    return server


def read_websocket_config(path: str):
    df = read_csv(path, header=None, index_col=0)
    return df.to_dict()





if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="id of worker")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    start_webserver(args.id, args.port)
