import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torchlib.websocket_utils import read_websocket_config

if __name__ == "__main__":
    from subprocess import Popen
    from time import sleep

    # worker_dict = {"alice": 8777, "bob": 8778, "charlie": 8779}
    worker_dict = read_websocket_config("configs/websetting/config.csv")
    print(worker_dict)

    worker_calls = [
        [
            "python",
            "torchlib/websocket_utils.py",
            "--port",
            str(id_dict["port"]),
            "--id",
            id_dict["id"],
            "--host",
            id_dict["host"],
        ]
        for row, id_dict in worker_dict.items()
    ]
    for call in worker_calls:
        Popen(call)
    print("started websockets")
