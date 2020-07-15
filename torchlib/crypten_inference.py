import torch
import crypten
from math import ceil
from time import time
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import classification_report

import sys, os.path

sys.path.insert(0, os.path.split(sys.path[0])[0])  # TODO: make prettier
from torchlib.models import resnet18

torch.set_num_threads(1)  # pylint:disable=no-member
workers = {"ALICE": 0, "BOB": 1}


@crypten.mpc.run_multiprocess(world_size=2)
def encrypt_model_and_data(
    model_path: str, batch_size: int, data_path: str, label_path: str, num_samples: int
):
    model = resnet18(
        pretrained=True,
        num_classes=3,
        in_channels=3,
        pooling="max",
        adptpool=False,
        input_size=224,
    )
    state = torch.load(model_path, map_location="cpu")
    dummy_input = torch.empty((1, 3, 224, 224))  # pylint:disable=no-member
    model.load_state_dict(state["model_state_dict"])
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=workers["ALICE"])
    print("Model successfully encrypted:", private_model.encrypted)

    data = crypten.load(data_path, src=workers["BOB"])
    targets = torch.load(label_path)  # , src=workers["BOB"])

    if num_samples == None:
        num_samples = data.shape[0]
    data = data[:num_samples]
    targets = targets[:num_samples]

    # Classify the encrypted data
    private_model.eval()
    predictions = []
    t = time()
    batch_num = ceil(num_samples / batch_size)
    for i in tqdm(range(batch_num), total=batch_num, leave=False, desc="Testing"):
        output_enc = private_model(data[i * batch_size : (i + 1) * batch_size])
        output = output_enc.get_plain_text()
        predictions.append(output.argmax(dim=1).flatten())
    t = time() - t
    # Compute the accuracy
    predictions = torch.cat(predictions)  # pylint:disable=no-member
    print(classification_report(targets, predictions))
    # print("\tAccuracy: {0:.4f}".format(accuracy.item(),))
    print("\tTotal time: {:.1f}\tTime per sample: {:.3f}".format(t, (t / num_samples)))


if __name__ == "__main__":
    crypten.init()
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images processed concurrently",
    )
    parser.add_argument(
        "--max_num_samples", type=int, default=None, help="stop after n samples"
    )
    parser.add_argument(
        "--model_weights", type=str, required=True, help="path to trained resnet-18"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/testdata.pt",
        help="path to stored data (torch tensor stored by torch.save)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/testlabels.pt",
        help="path to stored labels (torch tensor stored by torch.save)",
    )
    args = parser.parse_args()
    encrypt_model_and_data(
        args.model_weights,
        args.batch_size,
        args.data,
        args.labels,
        args.max_num_samples,
    )

