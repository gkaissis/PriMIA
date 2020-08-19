import argparse
import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torchlib.websocket_utils import read_websocket_config

if __name__ == "__main__":
    from subprocess import Popen
    from time import sleep
    import signal

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data folder of pneumonia data. Each worker has its subfolder called worker<i> where i is its index, and three subfolders with the classes.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config",
    )
    args = parser.parse_args()
    worker_dict = read_websocket_config("configs/websetting/config.csv")
    print(worker_dict)
    if args.data_dir == "mnist":
        worker_calls = [
            [
                "python",
                "-m",
                "Node",
                "--port",
                str(id_dict["port"]),
                "--id",
                id_dict["id"],
                "--data_directory",
                "mnist",
                "--config",
                args.config,
                "--host",
                "127.0.0.1",
            ]
            for row, id_dict in worker_dict.items()
        ]
    else:
        assert os.path.isdir(args.data_dir), "given path is no directory"
        worker_calls = [
            [
                "python",
                "-m",
                "Node",
                "--port",
                str(id_dict["port"]),
                "--id",
                id_dict["id"],
                "--host",
                "127.0.0.1",
                "--data_directory",
                os.path.join(args.data_dir, "worker{:d}/".format(i + 1)),
                "--config",
                args.config,
            ]
            for i, (row, id_dict) in enumerate(worker_dict.items())
            if id_dict["id"] != "crypto_provider"
        ]
        cp = [
            id_dict
            for id_dict in worker_dict.values()
            if id_dict["id"] == "crypto_provider"
        ]
        if len(cp) == 1:
            cp = cp[0]
            worker_calls.append(
                [
                    "python",
                    "-m",
                    "Node",
                    "--port",
                    str(cp["port"]),
                    "--id",
                    cp["id"],
                    "--host",
                    "127.0.0.1",
                ]
            )
    calls = []
    for call in worker_calls:
        calls.append(Popen(call))
    print("started websockets")

    def signal_handler(sig, frame):
        print("You pressed Ctrl+C!")
        for p in calls:
            print("terminate")
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    signal.pause()
