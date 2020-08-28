"""Convenience script to perform an inference benchmark using CrypTen. Should be run after create_crypten_data.py has generated a test dataset and corresponding labels. Should not be modified.
Returns classification metrics and whether the encrypted model performs identically as the plain-text model.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch

# pylint:disable=no-member
import crypten
from math import ceil
from time import time
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import classification_report

from sys import path
from os.path import split

path.insert(0, split(path[0])[0])
from torchlib.models import resnet18

workers = {"ALICE": 0, "BOB": 1}


@crypten.mpc.run_multiprocess(world_size=2)
def encrypt_model_and_data(
    model_path: str,
    batch_size: int,
    data_path: str,
    label_path: str,
    num_samples: int,
    start_at: int,
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

    data = torch.load(data_path)
    data_encrypted = crypten.load(data_path, src=workers["BOB"])
    targets = torch.load(label_path)  # , src=workers["BOB"])

    if num_samples == None:
        num_samples = data_encrypted.shape[0]
    data = data[start_at:num_samples]
    data_encrypted = data_encrypted[start_at:num_samples]
    targets = targets[start_at:num_samples]
    if data_encrypted.shape[0] == 0:
        raise ValueError(
            "Data is empty - Either start at too high or num samples too low"
        )

    # Classify the encrypted data
    private_model.eval()
    model.eval()
    predictions = []
    predictions_encrypted = []
    t = time()
    batch_num = ceil(num_samples / batch_size)
    start_at = ceil(start_at / batch_size)
    not_equal = 0
    try:
        with torch.no_grad():
            for i in tqdm(
                range(batch_num - start_at),
                total=batch_num,
                leave=False,
                desc="Testing",
                initial=start_at,
            ):
                output_enc = private_model(
                    data_encrypted[i * batch_size : (i + 1) * batch_size]
                )
                output = output_enc.get_plain_text()
                output = output.argmax(dim=1).flatten()
                predictions_encrypted.append(output)
                output_unencrypted = model(data[i * batch_size : (i + 1) * batch_size])
                output_unencrypted = output_unencrypted.argmax(dim=1).flatten()
                predictions.append(output_unencrypted)
                if not torch.equal(  # pylint:disable=no-member
                    output, output_unencrypted
                ):
                    not_equal += 1
                    print("Prediction of encrypted model not equal to plaintext model")
    except KeyboardInterrupt:
        print("Interrupted - stopping inference")
    t = time() - t
    if len(predictions_encrypted) == 0:
        exit()
    # Compute the accuracy
    predictions_encrypted = torch.cat(predictions_encrypted)  # pylint:disable=no-member
    print(
        classification_report(
            targets[: predictions_encrypted.shape[0]], predictions_encrypted
        )
    )
    # print("\tAccuracy: {0:.4f}".format(accuracy.item(),))
    print(
        "\tTotal time: {:.1f}\tTime per sample: {:.3f}".format(
            t, (t / predictions_encrypted.shape[0])
        )
    )
    print(
        "\tHad {:s} deviating predictions to plaintext model".format(
            "no" if not_equal == 0 else str(not_equal)
        )
    )
    if not_equal > 0:
        print("Stats of unencrypted model")
        predictions = torch.cat(predictions)  # pylint:disable=no-member
        print(classification_report(targets[: predictions.shape[0]], predictions))


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
    parser.add_argument(
        "--start_at", type=int, default=0, help="start inference at data sample i"
    )
    args = parser.parse_args()
    encrypt_model_and_data(
        args.model_weights,
        args.batch_size,
        args.data,
        args.labels,
        args.max_num_samples,
        args.start_at,
    )

