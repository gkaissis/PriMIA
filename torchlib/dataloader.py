import os
import random
import syft as sy
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import (  # pylint:disable=no-name-in-module
    manual_seed,
    stack,
    cat,
    std_mean,
    save,
    is_tensor,
    from_numpy,
    randperm,
    default_generator,
)
from torch._utils import _accumulate
import albumentations as a
from copy import deepcopy
from torch.utils import data as torchdata
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from os.path import splitext
from typing import Dict, Union, Set, Callable

from pathlib import Path
from .dicomtools import DicomLoader


class AlbumentationsTorchTransform:
    def __init__(self, transform, **kwargs):
        # print("init albu transform wrapper")
        self.transform = transform
        self.kwargs = kwargs

    def __call__(self, img):
        # print("call albu transform wrapper")
        if Image.isImageType(img):
            img = np.array(img)
        elif is_tensor(img):
            img = img.cpu().numpy()
        img = self.transform(image=img, **self.kwargs)["image"]
        # if img.max() > 1:
        #     img = a.augmentations.functional.to_float(img, max_value=255)
        img = from_numpy(img)
        if img.shape[-1] < img.shape[0]:
            img = img.permute(2, 0, 1)
        return img


class CombinedLoader:
    """Class that combines several data loaders and their extensions.

    Args: 
        mapping (Dict): Dictionary that maps loader names to tuples 
                        consisting of (corresponding extensions, loader method)
    """

    def __init__(
        self,
        mapping: Dict[str, Dict[str, Union[Set[str], Callable]]] = {
            "default": {
                "extensions": {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".ppm",
                    ".bmp",
                    ".pgm",
                    ".tif",
                    ".tiff",
                    ".webp",
                },
                "loader": default_loader,
            },
            "dicom": {"extensions": {".dcm", ".dicom"}, "loader": DicomLoader(3)},
        },
    ):
        self.extensions = set()
        self.mapping = mapping
        self.ext_to_loader_name = dict()
        for loader_name, defining_dict in mapping.items():
            self.extensions |= defining_dict["extensions"]
            for ext in defining_dict["extensions"]:
                if ext in self.ext_to_loader_name:
                    raise RuntimeError(
                        "Extension {:s} was passed for multiple loaders".format(ext)
                    )
                self.ext_to_loader_name[ext] = loader_name

    def __call__(self, path: Path, **kwargs):
        """Apply loader to path

        Args:
            path (Path): path to file.
            kwargs: kwargs passed to load methods

        Returns:
            Image: a PIL image of the given path

        Raises:
            RuntimeError: If loader for path extension not specified.
        """
        file_ending = splitext(path)[1].lower()
        if file_ending in self.extensions:
            return self.mapping[self.ext_to_loader_name[file_ending]]["loader"](
                path, **kwargs
            )
        else:
            raise RuntimeError(
                "file extension does not match specified supported extensions. "
                "Please provide the matching loader for the {:s} extension.".format(
                    file_ending
                )
            )

    def change_channels(self, num_channels: int):
        """Change the number of channels that are loaded (Default: 3)

        Args:
            num_channels (int): Number of channels. Currently only 1 and 3 supported

        Raises:
            RuntimeError: if num_channels is not 1 or 3
        """
        if num_channels not in [1, 3]:
            raise RuntimeError("Only 1 or 3 channels supported yet.")
        self.mapping["default"]["loader"] = (
            single_channel_loader if num_channels == 1 else default_loader
        )
        self.mapping["dicom"]["loader"] = DicomLoader(num_channels)


def create_albu_transform(args, mean, std):
    train_tf = transforms.RandomAffine(
        degrees=args.rotation,
        translate=(args.translate, args.translate),
        scale=(1.0 - args.scale, 1.0 + args.scale),
        shear=args.shear,
        #    fillcolor=0,
    )
    start_transformations = [
        a.Resize(args.inference_resolution, args.inference_resolution),
        a.RandomCrop(args.train_resolution, args.train_resolution),
    ]
    if args.clahe:
        start_transformations.extend(
            [
                a.FromFloat(dtype="uint8", max_value=1.0),
                a.CLAHE(always_apply=True, clip_limit=(1, 1)),
            ]
        )
    train_tf_albu = [
        a.VerticalFlip(p=args.individual_albu_probs),
    ]
    if args.randomgamma:
        train_tf_albu.append(a.RandomGamma(p=args.individual_albu_probs))
    if args.randombrightness:
        train_tf_albu.append(a.RandomBrightness(p=args.individual_albu_probs))
    if args.blur:
        train_tf_albu.append(a.Blur(p=args.individual_albu_probs))
    if args.elastic:
        train_tf_albu.append(a.ElasticTransform(p=args.individual_albu_probs))
    if args.optical_distortion:
        train_tf_albu.append(a.OpticalDistortion(p=args.individual_albu_probs))
    if args.grid_distortion:
        train_tf_albu.append(a.GridDistortion(p=args.individual_albu_probs))
    if args.grid_shuffle:
        train_tf_albu.append(a.RandomGridShuffle(p=args.individual_albu_probs))
    if args.hsv:
        train_tf_albu.append(a.HueSaturationValue(p=args.individual_albu_probs))
    if args.invert:
        train_tf_albu.append(a.InvertImg(p=args.individual_albu_probs))
    if args.cutout:
        train_tf_albu.append(
            a.Cutout(
                num_holes=5, max_h_size=80, max_w_size=80, p=args.individual_albu_probs
            )
        )
    if args.shadow:
        assert args.pretrained, "RandomShadows needs 3 channels"
        train_tf_albu.append(a.RandomShadow(p=args.individual_albu_probs))
    if args.fog:
        assert args.pretrained, "RandomFog needs 3 channels"
        train_tf_albu.append(a.RandomFog(p=args.individual_albu_probs))
    if args.sun_flare:
        assert args.pretrained, "RandomSunFlare needs 3 channels"
        train_tf_albu.append(a.RandomSunFlare(p=args.individual_albu_probs))
    if args.solarize:
        train_tf_albu.append(a.Solarize(p=args.individual_albu_probs))
    if args.equalize:
        train_tf_albu.append(a.Equalize(p=args.individual_albu_probs))
    if args.grid_dropout:
        train_tf_albu.append(a.GridDropout(p=args.individual_albu_probs))
    train_tf_albu.append(a.GaussNoise(var_limit=args.noise_std ** 2, p=args.noise_prob))
    end_transformations = [
        a.ToFloat(max_value=255.0),
        a.Normalize(mean, std, max_pixel_value=1.0),
    ]
    if not args.pretrained:
        end_transformations.append(
            a.Lambda(image=lambda x, **kwargs: x[:, :, np.newaxis])
        )
    train_tf_albu = AlbumentationsTorchTransform(
        a.Compose(
            [
                a.Compose(start_transformations),
                a.Compose(train_tf_albu, p=args.albu_prob),
                a.Compose(end_transformations),
            ]
        )
    )
    return transforms.Compose([train_tf, train_tf_albu,])


def calc_mean_std(dataset, save_folder=None):
    """
    Calculates the mean and standard deviation of `dataset` and
    saves them to `save_folder`.

    Needs a dataset where all images have the same size
    """
    accumulated_data = []
    for d in tqdm(
        dataset, total=len(dataset), leave=False, desc="accumulate data in dataset"
    ):
        if type(d) is tuple or type(d) is list:
            d = d[0]
        accumulated_data.append(d)
    if isinstance(dataset, torchdata.Dataset):
        accumulated_data = stack(accumulated_data)
    elif isinstance(dataset, torchdata.DataLoader):
        accumulated_data = cat(accumulated_data)
    else:
        raise NotImplementedError("don't know how to process this data input class")
    if accumulated_data.shape[1] in [1, 3]:  # ugly hack
        dims = (0, *range(2, len(accumulated_data.shape)))
    else:
        dims = (*range(len(accumulated_data.shape)),)
    std, mean = std_mean(accumulated_data, dim=dims)
    if save_folder:
        save(stack([mean, std]), os.path.join(save_folder, "mean_std.pt"))
    return mean, std


def single_channel_loader(filename):
    """Converts `filename` to a grayscale PIL Image
    """
    with open(filename, "rb") as f:
        img = Image.open(f).convert("L")
        return img.copy()


class LabelMNIST(MNIST):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        indices = np.isin(self.targets, labels).astype("bool")
        self.data = self.data[indices]
        self.targets = self.targets[indices]


class PathDataset(torchdata.Dataset):
    def __init__(
        self,
        root,
        transform=None,
        loader=CombinedLoader(),
        extensions=[
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
            ".dcm",
            ".dicom",
        ],
    ):
        super(PathDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = [
            f
            for f in os.listdir(root)
            if os.path.splitext(f)[1].lower() in extensions
            and not os.path.split(f)[1].lower().startswith("._")
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = self.loader(os.path.join(self.root, img_path))
        if self.transform:
            img = self.transform(img)
        return img


class RemoteTensorDataset(torchdata.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        return self.tensor[idx].copy()


class ImageFolderFromCSV(torchdata.Dataset):
    def __init__(
        self, csv_path, img_folder_path, transform=None, target_transform=None
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.img_folder_path = img_folder_path
        self.img_files = [
            i for i in os.listdir(img_folder_path) if not i.startswith(".")
        ]

        metastats = pd.read_csv(csv_path)

        metastats["class_label"] = metastats.apply(
            ImageFolderFromCSV.__meta_to_class__, axis=1
        )
        self.categorize_dict = dict(
            zip(metastats.X_ray_image_name, metastats.class_label)
        )
        for img in self.img_files:
            assert (
                img in self.categorize_dict.keys()
            ), "img label not known {:s}".format(str(img))
            if self.categorize_dict[img] == -1:
                self.img_files.remove(img)
                print("Ignore image {:s} because category is certain".format(img))

    @staticmethod
    def __meta_to_class__(row):
        if row["Label"] == "Normal":
            return 0
        if row["Label"] == "Pnemonia":  # i know this is a typo but was in original csv
            if row["Label_1_Virus_category"] == "bacteria":
                return 1
            if row["Label_1_Virus_category"] == "Virus":
                return 2
        return -1

    def __getitem__(self, i):
        img_path = self.img_files[i]
        label = self.categorize_dict[img_path]
        img = single_channel_loader(os.path.join(self.img_folder_path, img_path))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.img_files)


class PPPP(torchdata.Dataset):
    def __init__(
        self, label_path="data/Labels.csv", train=False, transform=None, seed=1,
    ):
        super().__init__()
        random.seed(seed)
        manual_seed(seed)
        self.train = train
        self.labels = pd.read_csv(label_path)
        self.labels = self.labels[
            self.labels["Dataset_type"] == ("TRAIN" if train else "TEST")
        ]
        self.transform = transform
        """
        Split into train and validation set
        if self.train:
            indices = [
                i
                for i in range(len(self.labels))
                if ((i % self.val_split) != 0 and self.val)
                or (not self.val and (i % self.val_split) == 0)
            ]
            self.labels = self.labels.drop(index=indices)
        """

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        label = row["Numeric_Label"]
        path = "train" if self.train else "test"
        path = os.path.join("data", path, row["X_ray_image_name"])
        img = single_channel_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    # def get_class_name(self, numeric_label):
    #    return self.class_names[numeric_label]

    """
    Works only if not torch.utils.torchdata.random_split is applied
    """

    def get_class_occurances(self):
        return dict(self.labels["Numeric_Label"].value_counts())

    def __compute_mean_std__(self):

        calc_mean_std(
            self, save_folder="data",
        )


##This is from torch.data.utils and adapted for our purposes
class Subset(torchdata.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = deepcopy(dataset)
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths, generator=default_generator):
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]

"""
    Data utility functions for the Medical Segmentation Decathlon
    http://medicaldecathlon.com/#tasks
    Based on the 'prepare_data' function in data_loader.py from M. Knolle 
    (TUM-AIMED/MoNet)
"""

import os
import sys
import gc
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import cv2 as cv
import math
from skimage.transform import resize
from nilearn.image import resample_img
from pathlib import Path, WindowsPath, PureWindowsPath, PosixPath, PurePosixPath
from tqdm import tqdm

# additional utility functions from original function 

def scale_array(array: np.array) -> np.array:
    """ Scales a numpy array from 0 to 1. Works in 3D 
        Return np.array """
    assert array.max() - array.min() > 0

    return ((array - array.min()) / (array.max() - array.min())).astype(np.float32)


def preprocess_scan(scan) -> np.array:
    """ Performs Preprocessing: (1) clips vales to -150 to 200, (2) scales values to lie in between 0 and 1, 
        (3) peforms rotations and flipping to move patient into referen position 
        Return: np.array """
    scan = np.clip(scan, -150, 200)
    scan = scale_array(scan)
    scan = np.rot90(scan)
    scan = np.fliplr(scan)

    return scan

def rotate_label(label_volume) -> np.array:
    """ Rotates and flips the label in the same way the scans were rotated and flipped
        Return: np.array """

    label_volume = np.rot90(label_volume)
    label_volume = np.fliplr(label_volume)

    return label_volume.astype(np.float32)

def preprocess_and_convert_to_numpy(
    nifti_scan: nib.Nifti1Image, nifti_mask: nib.Nifti1Image
) -> list:
    """ Convert scan and label to numpy arrays and perform preprocessing
            Return: Tuple(np.array, np.array)  """
    np_scan = nifti_scan.get_fdata()
    np_label = nifti_mask.get_fdata()
    nifti_mask.uncache()
    nifti_scan.uncache()
    np_scan = preprocess_scan(np_scan)
    np_label = rotate_label(np_label)
    assert np_scan.shape == np_label.shape

    return np_scan, np_label

def get_name(nifti: nib.Nifti1Image, path: Path) -> str:
    "Gets the original filename of a Nifti."
    file_dir = nifti.get_filename()
    if isinstance(path, PosixPath) or isinstance(path, PurePosixPath):
        file_name = file_dir.split("/")[-1]
    else:
        file_name = file_dir.split("\\")[-1]
    return file_name

def merge_labels(label_volume: np.array) -> np.array:
    """ Merges Tumor and Pancreas labels into one volume with background=0, pancreas & tumor = 1
        Input: label_volume = 3D numpy array
        Return: Merged Label volume """
    merged_label = np.zeros(label_volume.shape, dtype=np.float32)
    merged_label[label_volume == 1] = 1.
    merged_label[label_volume == 2] = 1.
    return merged_label

def bbox_dim_3D(img: np.array):
    """Finds the corresponding dimensions for the 3D Bounding Box from the Segmentation Label volume
       Input: img = 3D numpy array, 
       Return: row-min, row-max, collumn-min, collumn-max, z-min, z_max """
    if np.nonzero(img)[0].size == 0:
        print("Warning Empty Label Mask")
        return None
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def create_2D_label(rmin: int, rmax: int, cmin: int, cmax: int, res: int) -> np.array:
    """Creates a 2D Label from a Bounding Box 
        Input: rmin, rmax, cmin, cmax = dimensions of Bounding Box, res = resolution of required label
        Return: 2D numpy array """
    label = np.zeros((res, res), dtype=np.float32)
    label[rmin:rmax, cmin:cmax] = 1
    return label

def create_3D_label(
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    zmin: int,
    zmax: int,
    res: int,
    num_slices: int,
) -> np.array:
    """Creates a 3D Label from a Bounding Box 
        Input: rmin, rmax, cmin, cmax, zmin, zmax = dimensions of Bounding Box, res = resolution of required label
        Return: 3D numpy array """
    label_volume = np.zeros((res, res, num_slices), dtype=np.float32)
    label_volume[rmin:rmax, cmin:cmax, zmin:zmax] = 1.0
    return label_volume

def convert_to_2d(arr: np.array) -> np.array:
    """Takes a 4d array of shape: (num_samples, res_x, res_y, sample_height) and destacks the 3d scans 
        converting it to a 3d array of shape: (num_samples*sample_height, res_x, res_y)"""
    assert len(arr.shape) == 4
    reshaped = np.concatenate(arr, axis=-1)
    assert len(reshaped.shape) == 3
    return np.moveaxis(reshaped, -1, 0)

def crop_volume(
    data_volume: np.array, label_volume: np.array, crop_height: int = 32
) -> np.array:
    """Crops two 3D Numpy array along the zaxis to crop_height. Finds the midpoint of the pancreas label along the z-axis and crops [..., zmiddle-(crop_height//2):zmiddle+(crop_height//2)].
       Return: two np.array s """
    rmin, rmax, cmin, cmax, zmin, zmax = bbox_dim_3D(label_volume)

    zmiddle = (zmin + zmax) // 2
    z_min = zmiddle - (crop_height // 2)
    if z_min < 0:
        z_min = 0
    z_max = zmiddle + (crop_height // 2)
    cropped_data_volume = data_volume[:, :, z_min:z_max]
    cropped_label_volume = label_volume[:, :, z_min:z_max]

    return cropped_data_volume, cropped_label_volume

class MSD_data(torchdata.Dataset):

    def __init__(
        self, 
        path_string: str,
        res: int,
        res_z: int,
        sample_limit: int = -1,  # 281 for the whole dataset
        crop_height: int = 32,
        mode="2D",
        label_mode="seg",
        mrg_labels: bool = True,
        ):
        self.path_string = path_string
        self.res = res
        self.res_z = res_z 
        self.sample_limit = sample_limit
        self.crop_height = crop_height
        self.mode = mode
        self.label_mode = label_mode
        self.mrg_labels = mrg_labels

        # as in original function 
        assert (crop_height % 16) == 0
        if label_mode == "bbox-coord":
                raise NotImplementedError()

        # dynamically add search for all *.nii.* files that are 
        # present in the subfolders of **/data_path/imagesTr
        # and check if for every scan there exists one label 
        scan_path = Path(path_string) / "imagesTr"
        assert scan_path.exists() # as in original function 
        scan_names = [
            # TODO: Solve problem: don't want to load in all images but how do I call get_names then? 
            #       Is what I'm doing here also valid or not the real file names? NO - assert doesn't pass
            #       What is the difference? 
            #get_name(file, scan_path) for i, file in enumerate(scan_path.rglob("*.nii.*")) if i < sample_limit 
            file for i, file in enumerate(scan_path.rglob("*.nii.*")) if i < sample_limit 
        ]

        label_path = Path(path_string) / "labelsTr"
        assert label_path.exists() # as in original function 
        label_names = [
            #get_name(file, label_path) for i, file in enumerate(label_path.rglob("*.nii.*")) if i < sample_limit
            file for i, file in enumerate(label_path.rglob("*.nii.*")) if i < sample_limit
        ]

        # Make sure that for each scan there exists a label 
        # (the labels have the same name as the scans
        #  there just stored in another folder)
        #TODO: Change with get_name
        #assert scan_names==label_names

        self.scan_names = scan_names 
        self.label_names = label_names 

    #TODO: Possible change that? The same way as in Moritzes version? 
    def __len__(self):
        """
            If we don't specify how many samples we want we get all samples
            that are present in the directories (see in __init__())
        """
        return len(self.scan_names) if self.sample_limit == -1 else self.sample_limit

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def get_item_from_index(self, index):
        scan_id = self.scan_names[index] 
        label_id = self.label_names[index]

        scan = nib.load(str(scan_id))
        label = nib.load(str(label_id))

        # preprocessing scan and label, extracting np array data
        scan, label = preprocess_and_convert_to_numpy(scan, label)

        # merging tumor and pancreas labels in the label mask
        if self.mrg_labels:
            label = merge_labels(label)

        # finding bounding box from segmentation label
        if self.label_mode == "bbox-seg" or self.label_mode == "bbox-coord":
            b = bbox_dim_3D(label)
            if b == None:
                print(
                    "Couldn't generate a bounding box for the label in scan:", scan_id
                )
            # creating label mask from bounding box dimensions
            label = create_3D_label(
                b[0], b[1], b[2], b[3], b[4], b[5], label.shape[1], label.shape[2],
            )

        # cropping scan and label volumes to reduce the number of non-pancreas slices
        cropped_scan, cropped_label = crop_volume(scan, label, crop_height=self.crop_height)
        assert cropped_scan.shape == cropped_label.shape

        scan = resize(
            cropped_scan, (self.res, self.res, self.res_z), preserve_range=True, order=1
        ).astype(np.float32)
        label = resize(
            cropped_label, (self.res, self.res, self.res_z), preserve_range=True, order=0
        ).astype(np.uint8)

        if self.mode == "2D":
            #print("... converting data to 2D slices")
            # convert_to_2d expects an array of shape: num_samples, xres, yres, height 
            # will return num_samples*height, xres, yres
            scan = np.expand_dims(scan, 0)
            label = np.expand_dims(label, 0)
            scan, label = (
                convert_to_2d(scan),
                convert_to_2d(label),
            )
            # we want each of the new slices to be interpreted as separate sample
            # we shift the slices to the number of samples and introduce an empty channel-dim
            # MoNet expects samples of batch_size x channel (=1) x 2D_image_dim
            scan = np.expand_dims(scan, 1)
            #label = np.expand_dims(label, 1)

        # convert to tensors 
        return from_numpy(scan.copy()), from_numpy(label.copy()).long()

"""
    MSD dataset as normal image-dataset. Assumes dataset was already preprocessed. 
"""

from os import listdir
from os.path import isfile, join

class MSD_data_images(torchdata.Dataset): 
    def __init__(
        self, 
        img_path="./data",
        ):
    
        self.input_path = img_path + "/inputs/"
        self.target_path = img_path + "/labels/"

        assert os.path.exists(self.input_path) 
        scan_names = [
            file for file in listdir(self.input_path) if isfile(join(self.input_path, file))
        ]

        assert os.path.exists(self.target_path) 
        label_names = [
            file for file in listdir(self.target_path) if isfile(join(self.target_path, file))
        ]

        # check that for each scan there exists a label
        # sort important in-case files are not structured the same way (problem in Colab e.g.)
        scan_names.sort()
        label_names.sort()
        assert scan_names==label_names

        self.scan_names = scan_names 
        self.label_names = label_names 

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def get_item_from_index(self, index):
        # because images are named from 0.jpg, 1.jpg, ...
        scan_path = self.input_path + f'{index}.jpg'
        label_path = self.target_path + f'{index}.jpg'

        scan_img = Image.open(scan_path)
        label_img = Image.open(label_path)

        scan_np = np.array(scan_img, dtype=np.float32) #float
        label_np = np.array(label_img, dtype=np.uint8) #int

        # MoNet expects a tensor of shape channel x xres x yres 
        # channel = 1
        scan_np = np.expand_dims(scan_np, axis=0)

        scan = from_numpy(scan_np)
        # BCELoss expects float tensors and not byte tensors
        label = from_numpy(label_np).float()
        
        # no transforms necessary, because all already done in preprocessing

        return scan, label

"""
    Data utility functions from I2DL class - N.Remerscheid
""" 

import _pickle as pickle

# pylint: disable=C0326
SEG_LABELS_LIST = [
    {"id": -1, "name": "void",       "rgb_values": [0,   0,    0]},
    {"id": 0,  "name": "building",   "rgb_values": [128, 0,    0]},
    {"id": 1,  "name": "grass",      "rgb_values": [0,   128,  0]},
    {"id": 2,  "name": "tree",       "rgb_values": [128, 128,  0]},
    {"id": 3,  "name": "cow",        "rgb_values": [0,   0,    128]},
    {"id": 4,  "name": "horse",      "rgb_values": [128, 0,    128]},
    {"id": 5,  "name": "sheep",      "rgb_values": [0,   128,  128]},
    {"id": 6,  "name": "sky",        "rgb_values": [128, 128,  128]},
    {"id": 7,  "name": "mountain",   "rgb_values": [64,  0,    0]},
    {"id": 8,  "name": "airplane",   "rgb_values": [192, 0,    0]},
    {"id": 9,  "name": "water",      "rgb_values": [64,  128,  0]},
    {"id": 10, "name": "face",       "rgb_values": [192, 128,  0]},
    {"id": 11, "name": "car",        "rgb_values": [64,  0,    128]},
    {"id": 12, "name": "bicycle",    "rgb_values": [192, 0,    128]},
    {"id": 13, "name": "flower",     "rgb_values": [64,  128,  128]},
    {"id": 14, "name": "sign",       "rgb_values": [192, 128,  128]},
    {"id": 15, "name": "bird",       "rgb_values": [0,   64,   0]},
    {"id": 16, "name": "book",       "rgb_values": [128, 64,   0]},
    {"id": 17, "name": "chair",      "rgb_values": [0,   192,  0]},
    {"id": 18, "name": "road",       "rgb_values": [128, 64,   128]},
    {"id": 19, "name": "cat",        "rgb_values": [0,   192,  128]},
    {"id": 20, "name": "dog",        "rgb_values": [128, 192,  128]},
    {"id": 21, "name": "body",       "rgb_values": [64,  64,   0]},
    {"id": 22, "name": "boat",       "rgb_values": [192, 64,   0]}]


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


class SegmentationData(torchdata.Dataset):

    def __init__(self, image_paths_file):
        self.root_dir_name = os.path.dirname(image_paths_file)

        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_id = self.image_names[index].replace('.bmp', '')

        img = Image.open(os.path.join(self.root_dir_name,
                                      'images',
                                      img_id + '.bmp')).convert('RGB')
        center_crop = transforms.CenterCrop(240)
        # TODO: TEMP.
        #center_crop = transforms.CenterCrop(240)
        #img = center_crop(img)
        #img = to_tensor(img)

        # TODO: TEMP. (only one channel -> for testing)
        #img = img[:1, :, :]

        target = Image.open(os.path.join(self.root_dir_name,
                                         'targets',
                                         img_id + '_GT.bmp'))
        target = center_crop(target)
        target = np.array(target, dtype=np.int64)

        target_labels = target[..., 0]
        for label in SEG_LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis=2)
            target_labels[mask] = label['id']

        target_labels = from_numpy(target_labels.copy())

        return img, target_labels

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import sys
    from tqdm import tqdm
    import numpy as np

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    )
    from torchlib.utils import AddGaussianNoise

    ds = PPPP(train=True, transform=transforms.ToTensor())
    print("Class distribution")
    print(ds.get_class_occurances())

    sizes = []

    for data, _ in tqdm(ds, total=len(ds), leave=False):
        sizes.append(data.size()[1:])
    sizes = np.array(sizes)
    print(
        "data resolution stats: \n\tmin: {:s}\n\tmax: {:s}\n\tmean: {:s}\n\tmedian: {:s}".format(
            str(np.min(sizes, axis=0)),
            str(np.max(sizes, axis=0)),
            str(np.mean(sizes, axis=0)),
            str(np.median(sizes, axis=0)),
        )
    )

    ds = PPPP(train=False)

    L = len(ds)
    print("length test set: {:d}".format(L))
    img, label = ds[1]
    img.show()

    tf = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),]
    )
    ds = PPPP(train=True, transform=tf)

    ds.__compute_mean_std__()
    L = len(ds)
    print("length train set: {:d}".format(L))

    from matplotlib import pyplot as plt

    ds = PPPP()
    hist = ds.labels.hist(bins=3, column="Numeric_Label")
    plt.show()
