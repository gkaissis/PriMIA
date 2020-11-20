# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import nibabel as nib
from dipy.align.reslice import reslice
from pathlib import Path, WindowsPath, PureWindowsPath, PosixPath, PurePosixPath
from tqdm import tqdm
import os
from typing import Tuple
import numpy as np
from skimage.transform import resize


f_string = "C:\\Users\\Moritz\\repositories\\dataset\\Task07_Pancreas"
#f_string = "/Users/moritzknolle/Desktop/Uni/deep_learning_adenocarcinoma/Data/Task07_Pancreas"
data_path = Path(f_string)

img_path = data_path / "imagesTr"
label_path = data_path / "labelsTr"

assert img_path.exists() and label_path.exists()

# loading scans
scan_names = [file for file in img_path.rglob("*.nii.*")]
scans = [nib.load(str(scan_name)) for scan_name in scan_names]

# loading labels
label_names = [file for file in label_path.rglob("*.nii.*")]
labels = [nib.load(str(scan_name)) for scan_name in label_names]

assert len(labels) == len(scans)

if not os.path.exists(data_path / "images_resampled") and not os.path.exists(data_path / "labels_resampled"):
    imgs_path = data_path / "images_resampled"
    lbls_path = data_path / "labels_resampled"
    imgs_path.mkdir()
    lbls_path.mkdir()

NEW_ZOOMS = (1., 1., 3.)


def get_name(nifti:nib.Nifti1Image)->str:
    "Gets the original filename of a Nifti."
    file_dir = nifti.get_filename()
    if isinstance(data_path, PosixPath) or isinstance(data_path, PurePosixPath):
        file_name = file_dir.split("/")[-1]
    else:
        file_name = file_dir.split("\\")[-1]
    return file_name


def resample_nii(nifti:nib.Nifti1Image, new_pix_dims:Tuple[float,...], num_workers:int) -> Tuple[np.ndarray, ...]:
    "Resamples nifti pixels to new_pix_dims and returns the new array and new affine using num_workers threads."
    return reslice(nifti.get_data(), nifti.affine, nifti.header.get_zooms(), new_pix_dims, num_processes=num_workers)


def resize_array(ar:np.ndarray, out_size:Tuple[int,...], isLabel:bool) -> np.ndarray:
    "Rescales ar to out_size with nearest neighbour and maintains value ranges."
    return resize(ar, out_size, preserve_range=True, order = 0)


i=0
for scan, label in zip(scans, labels):
    i+=1
    print("... resampling scan no:", i, get_name(scan))
    resampled_vol, new_vol_affine = resample_nii(scan, NEW_ZOOMS, 0)
    resampled_mask, new_mask_affine = resample_nii(label, NEW_ZOOMS, 0)

    resized_vol = resize_array(resampled_vol, (512, 512, resampled_vol.shape[2]), False)
    resized_mask = resize_array(resampled_mask, (512, 512, resampled_mask.shape[2]), True)

    vol_resliced = nib.Nifti1Image(resized_vol, new_vol_affine)
    mask_resliced = nib.Nifti1Image(resized_mask, new_mask_affine)

    vol_path = data_path / "images_resampled" / get_name(scan)
    mask_path = data_path / "labels_resampled" / get_name(label)
    nib.save(vol_resliced, str(vol_path))
    nib.save(mask_resliced, str(mask_path))
print("done")






