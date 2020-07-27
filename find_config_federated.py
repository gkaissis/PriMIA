from train import main
from argparse import Namespace
import random

random.seed(1)


if __name__ == "__main__":
    args = Namespace(
        batch_size=200,
        beta1=0.2896660431032566,
        beta2=0.9331401612981756,
        config="optuna",
        dataset="pneumonia",
        encrypted_inference=False,
        end_lr=4.5633571094235025e-06,
        epochs=25,
        inference_resolution=224,
        log_interval=10,
        lr=0.00020771896973198646,
        mixup=False,
        model="resnet-18",
        no_cuda=True,
        noise_prob=0.8749703191366571,
        noise_std=0.11311499988886073,
        optimizer="Adam",
        pooling_type="max",
        pretrained=True,
        restarts=1,
        rotation=7,
        scale=0.08767811048707483,
        seed=1,
        shear=14,
        test_batch_size=10,
        test_interval=1,
        train_federated=True,
        train_resolution=224,
        translate=0.0,
        validation_split=10,
        vertical_flip_prob=0.9974265445792014,
        visdom=False,
        websockets=False,
        weight_classes=True,
        weight_decay=3.4969641550458494e-12,
    )
    for i in range(10):
        args.repetitions_dataset=random.randrange(2, 6)
        args.sync_every_n_batch = 5
        args.wait_interval = 0.1
        args.keep_optim_dict = random.choice([True, False])
        args.weighted_averaging = random.choice([True, False])
        best_val_acc = main(args, verbose=True)
