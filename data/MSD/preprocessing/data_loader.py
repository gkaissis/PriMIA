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


def get_name(nifti: nib.Nifti1Image, path: Path) -> str:
    "Gets the original filename of a Nifti."
    file_dir = nifti.get_filename()
    if isinstance(path, PosixPath) or isinstance(path, PurePosixPath):
        file_name = file_dir.split("/")[-1]
    else:
        file_name = file_dir.split("\\")[-1]
    return file_name


def load_data(path: Path, num_samples: int):
    """ path needs to be a pathlib Path object. Loads in .NIFTI CT Volumes from a file path into a dictionary
        Return: Data Dictionary (Key: filename , Value: Niib Image)"""
    data_dict = {}
    if not path.exists() or not path.is_dir():
        raise ValueError(
            "Given path does not exist or is a file : " + str(path))
    file_names = [file for file in path.rglob("*.nii.*")]
    files = [
        nib.load(str(file)) for i, file in enumerate(file_names) if i < num_samples
    ]
    num_samples = len(files)
    data_dict = {get_name(file, path): file for file in files}
    return data_dict, num_samples


def merge_labels(label_volume: np.array) -> np.array:
    """ Merges Tumor and Pancreas labels into one volume with background=0, pancreas & tumor = 1
        Input: label_volume = 3D numpy array
        Return: Merged Label volume """
    merged_label = np.zeros(label_volume.shape, dtype=np.float32)
    merged_label[label_volume == 1] = 1.
    merged_label[label_volume == 2] = 1.
    return merged_label


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


def bbox_dim_2D(img: np.array):
    """Finds the corresponding dimensions for the 2D Bounding Box from the Segmentation Label Slice
       Input: img = 2D numpy array, 
       Return: row-min, row-max, collumn-min, collumn-max """
    if np.nonzero(img)[0].size == 0:
        return None
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


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


def convert_to_2d(arr: np.array) -> np.array:
    """Takes a 4d array of shape: (num_samples, res_x, res_y, sample_height) and destacks the 3d scans 
        converting it to a 3d array of shape: (num_samples*sample_height, res_x, res_y)"""
    assert len(arr.shape) == 4
    reshaped = np.concatenate(arr, axis=-1)
    assert len(reshaped.shape) == 3
    return np.moveaxis(reshaped, -1, 0)


def prepare_data(
    path_string: str,
    res: int,
    res_z: int,
    num_samples: int = 281,  # 281 for the whole dataset
    crop_height: int = 32,
    mode="2D",
    label_mode="seg",
    mrg_labels: bool = True,
):
    """ Prepares data for Training
        Params: 
            mode: 2D or 3D (return either 2D or 3D training tensors and labels) 
            res: resolution to downscale data to
            crop_height: crop region that determines the amount of context (non-label slices) that will be included in the label
            label_mode: seg -> segmentation; bbox-seg -> bounding box segmentation labels; bbox-coord
        Return: x_train, y_train, x_test, y_test """

    data_path = Path(path_string)

    img_path = data_path / "imagesTr"
    label_path = data_path / "labelsTr"

    assert img_path.exists() and label_path.exists()
    assert (crop_height % 16) == 0
    # importing training data
    print("... preparing data")
    print("... importing training data")
    scan_data, num_scans = load_data(img_path, num_samples)

    # importing labels
    print("... importing labels ")
    segmentation_labels, num_labels = load_data(label_path, num_samples)

    assert num_scans == num_labels

    print(f"... preparing training data and labels({label_mode})")
    #bar = Bar("... Processing", max=num_samples)
    data, labels = [], []
    for scan_name in tqdm(scan_data.keys()):
        # checking label and scan name match up
        if scan_name in segmentation_labels.keys():
            scan, label = scan_data[scan_name], segmentation_labels[scan_name]
        else:
            print("Couldn't find corresponding label for scan: ", scan_name)
            break

        # preprocessing scan and label, extracting np array data
        scan, label = preprocess_and_convert_to_numpy(scan, label)

        # merging tumor and pancreas labels in the label mask
        if mrg_labels:
            label = merge_labels(label)

        # finding bounding box from segmentation label
        if label_mode == "bbox-seg" or label_mode == "bbox-coord":
            b = bbox_dim_3D(label)
            if b == None:
                print(
                    "Couldn't generate a bounding box for the label in scan:", scan_name
                )
                break
                # creating label mask from bounding box dimensions
            label = create_3D_label(
                b[0], b[1], b[2], b[3], b[4], b[5], label.shape[1], label.shape[2],
            )

        # cropping scan and label volumes to reduce the number of non-pancreas slices
        cropped_scan, cropped_label = crop_volume(
            scan, label, crop_height=crop_height)
        assert cropped_scan.shape == cropped_label.shape
        scan = resize(
            cropped_scan, (res, res, res_z), preserve_range=True, order=1
        ).astype(np.float32)
        label = resize(
            cropped_label, (res, res, res_z), preserve_range=True, order=0
        ).astype(np.uint8)

        if label_mode == "bbox-coord":
            raise NotImplementedError()

        data.append(scan)
        labels.append(label)

    print("... total amount of training data:", len(data))
    print("... total amount of labels:", len(labels))

    # --------------------------------------------------------------------------------------------------------------

    # SPLITING INTO TRAIN AND TEST SET
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.3, shuffle=True, random_state=42,
    )
    print(
        "\ntrain set:",
        len(train_data),
        "scans and",
        len(train_labels),
        "label volumes",
    )
    print("test set:", len(test_data), "scans and",
          len(test_labels), "label volumes")

    # Splitting Training Set into partial training set and validation set
    (
        train_data_partial,
        train_data_val,
        train_labels_partial,
        train_labels_val,
    ) = train_test_split(train_data, train_labels, test_size=0.1, shuffle=True)
    print(
        "train set was split into partial train set:",
        len(train_data_partial),
        "scans and",
        len(train_labels_partial),
        "label volumes",
    )
    print(
        "validation set:",
        len(train_data_val),
        "scans and",
        len(train_labels_val),
        "label volumes",
    )

    # Garbage Collection
    data = None
    labels = None
    gc.collect()

    X_train_partial, y_train_partial = (
        np.array(train_data_partial),
        np.array(train_labels_partial),
    )
    X_val, y_val = np.array(train_data_val), np.array(train_labels_val)
    X_test, y_test = np.array(test_data), np.array(test_labels)

    if mode == "2D":
        print("... converting data to 2D slices")
        X_train_partial, y_train_partial = (
            convert_to_2d(X_train_partial),
            convert_to_2d(y_train_partial),
        )
        X_val, y_val = convert_to_2d(X_val), convert_to_2d(y_val)
        X_test, y_test = convert_to_2d(X_test), convert_to_2d(y_test)

    print("... finished preparing data")

    return X_train_partial, y_train_partial, X_val, y_val, X_test, y_test
