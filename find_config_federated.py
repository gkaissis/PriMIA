from train import main
from argparse import Namespace
import random
from numpy import random as rnd
from numpy import power

random.seed(1)
rnd.seed(1)


def lognuniform(low=0, high=1, size=None, base=10.0):
    return power(base, rnd.uniform(low, high, size))


if __name__ == "__main__":

    for i in range(30):
        args = Namespace(
            batch_size=200,
            config="optuna",
            dataset="pneumonia",
            encrypted_inference=False,
            inference_resolution=224,
            log_interval=10,
            model="resnet-18",
            no_cuda=True,
            optimizer="Adam",
            pooling_type="max",
            pretrained=True,
            seed=1,
            test_batch_size=10,
            test_interval=1,
            train_federated=True,
            train_resolution=224,
            translate=0.0,
            validation_split=10,
            visdom=False,
            websockets=False,
        )
        args.lr = lognuniform(1e-5, 1e-3)
        args.end_lr = lognuniform(args.lr, 1e-6)
        args.restarts = random.choice([0, 1])
        args.beta1 = rnd.uniform(0.25, 0.95)
        args.beta2 = rnd.uniform(0.9, 1.0)
        args.weight_decay = lognuniform(1e-12, 1e-3)
        args.weight_classes = random.choice([True, False])
        args.vertical_flip_prob = rnd.uniform(0.0, 1.0)
        args.rotation = random.randrange(0, 45)
        args.scale = rnd.uniform(0.0, 0.5)
        args.shear = random.randrange(0, 45)
        args.noise_std = rnd.uniform(0.0, 0.15)
        args.noise_prob = rnd.uniform(0.0, 1.0)
        args.mixup = random.choice([True, False])
        if args.mixup:
            args.mixup_lambda = random.choice(
                [0.1, 0.25, 0.49999, None]  # 0.5 breaks federated weight calculation
            )
            args.mixup_prob = rnd.uniform(0.0, 1.0)
        args.repetitions_dataset = random.randrange(3, 7)
        args.epochs = 25 // args.repetitions_dataset
        args.sync_every_n_batch = random.randrange(2, 10)
        args.wait_interval = 0.1
        args.keep_optim_dict = random.choice([True, False])
        args.weighted_averaging = random.choice([True, False])
        try:
            best_val_acc = main(args, verbose=True)
        except Exception as e:
            print("Exception {:s} with args:\n{:s}".format(str(e), str(args)))
