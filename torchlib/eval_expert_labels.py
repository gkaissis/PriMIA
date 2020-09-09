import torch
import pandas as pd
import albumentations as a
import argparse
import sys
import os.path
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torchlib.utils import stats_table
from torchlib.dataloader import AlbumentationsTorchTransform, CombinedLoader
from torchlib.models import vgg16, resnet18, conv_at_resolution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weights",
        default=None,
        help="if you want to evaluate model against expert label "
        "instead of expert against ground truth.",
    )
    cmd_args = parser.parse_args()

    expert_labels = pd.read_csv("data/expert_labels.csv")
    el_list = expert_labels["Einschätzung"].to_list()

    test_data = pd.read_csv("data/Labels.csv")
    test_data = test_data[test_data["Dataset_type"] == "TEST"]
    true_labels = test_data["Numeric_Label"].to_list()
    if cmd_args.model_weights:
        state = torch.load(cmd_args.model_weights)
        model_weights = state["model_state_dict"]
        args = state["args"]
        mean, std = state["val_mean_std"]
        loader = CombinedLoader()
        num_classes = 3
        tf = [
            a.Resize(args.inference_resolution, args.inference_resolution),
            a.CenterCrop(args.inference_resolution, args.inference_resolution),
        ]
        if hasattr(args, "clahe") and args.clahe:
            tf.append(a.CLAHE(always_apply=True, clip_limit=(1, 1)))
        tf.extend(
            [
                a.ToFloat(max_value=255.0),
                a.Normalize(
                    mean.cpu().numpy()[None, None, :],
                    std.cpu().numpy()[None, None, :],
                    max_pixel_value=1.0,
                ),
            ]
        )
        tf = AlbumentationsTorchTransform(a.Compose(tf))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.model == "vgg16":
            model = vgg16(
                pretrained=args.pretrained,
                num_classes=num_classes,
                in_channels=3 if args.pretrained else 1,
                adptpool=False,
                input_size=args.inference_resolution,
                pooling=args.pooling_type,
            )
        elif args.model == "simpleconv":
            if args.pretrained:
                raise RuntimeError("No pretrained version available")
            model = conv_at_resolution[args.train_resolution](
                num_classes=num_classes,
                in_channels=3 if args.pretrained else 1,
                pooling=args.pooling_type,
            )
        elif args.model == "resnet-18":
            model = resnet18(
                pretrained=args.pretrained,
                num_classes=num_classes,
                in_channels=3 if args.pretrained else 1,
                adptpool=False,
                input_size=args.inference_resolution,
                pooling=args.pooling_type if hasattr(args, "pooling_type") else "avg",
            )
        else:
            raise NotImplementedError("model unknown")
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        # test method
        model.eval()
        folder_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}
        convert_to_el_labels = {0: 1, 1: 0, 2: 2}
        labels = []
        for n, l in tqdm(
            zip(test_data["X_ray_image_name"], test_data["Numeric_Label"]),
            desc="classify images",
            leave=False,
        ):
            file_name = os.path.join("data", "test", folder_names[l], n)
            img = loader(file_name)
            img = tf(img).unsqueeze(0).to(device)
            pred = torch.argmax(model(img)).cpu().item()  # pylint:disable=no-member
            labels.append(convert_to_el_labels[pred])
        # print(
        #     stats_table(
        #         confusion_matrix(true_labels, labels),
        #         classification_report(
        #             true_labels, labels, output_dict=True, zero_division=0
        #         ),
        #         matthews_coeff=matthews_corrcoef(true_labels, labels),
        #         class_names=["normal", "bacterial", "viral"],
        #     )
        # )
        print(
            stats_table(
                confusion_matrix(el_list, labels),
                classification_report(
                    el_list, labels, output_dict=True, zero_division=0
                ),
                matthews_coeff=matthews_corrcoef(el_list, labels),
                class_names=["normal", "bacterial", "viral"],
            )
        )
    else:
        print(
            stats_table(
                confusion_matrix(true_labels, el_list),
                classification_report(
                    true_labels, el_list, output_dict=True, zero_division=0
                ),
                matthews_coeff=matthews_corrcoef(true_labels, el_list),
                class_names=["normal", "bacterial", "viral"],
            )
        )

