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
    parser.add_argument("--dataset", choices=["mnist", "pneumonia"], required=True)
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to data folder of pneumonia data. Each worker has its subfolder called workeri where i is its index and three subfolders with the classes.",
    )
    args = parser.parse_args()
    assert os.path.isdir(args.path), 'given path is no directory'
    # worker_dict = {"alice": 8777, "bob": 8778, "charlie": 8779}
    worker_dict = read_websocket_config("configs/websetting/config.csv")
    print(worker_dict)
    if args.dataset == "mnist":
        worker_calls = [
            [
                "python",
                "torchlib/websocket_utils.py",
                "--port",
                str(id_dict["port"]),
                "--id",
                id_dict["id"],
                "--data_directory",
                "mnist",
            ]
            for row, id_dict in worker_dict.items()
        ]
    elif args.dataset == "pneumonia":
        worker_calls = [
            [
                "python",
                "torchlib/websocket_utils.py",
                "--port",
                str(id_dict["port"]),
                "--id",
                id_dict["id"],
                "--data_directory",
                # "/home/alex/worker_emulation/all_samples",
                os.path.join(args.path, "worker{:d}/".format(i + 1)),
            ]
            for i, (row, id_dict) in enumerate(worker_dict.items())
        ]
    else:
        raise NotImplementedError("dataset not implemented")
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
