""" Interface to load DICOM files as numpy arrays or PIL Images.
Warnings:
- This module is tested and optimised only for paediatric chest X-Rays. 
  Other formats (e.g. CT) are unsupported.
- Despite extensive testing, we cannot guarantee that this will 
  work for every DICOM in the world. 
  DICOM is a very complex format with a myriad edge cases.
- Using DICOMS may not even be the best idea: If the images are highly 
  unstandardised (e.g. complex supine radiographs), it may be a better 
  idea to convert them and manually crop them to get better algorithm predictions.

Documentation for `load_dcm` partially adapted from tensorflow-io.
Citation for the dicom loader package used:
@misc{marcelo_lerendegui_2019_3337331,
    author       = {Marcelo Lerendegui and Ouwen Huang},
    title        = {Tensorflow Dicom Decoder},
    month        = jul,
    year         = 2019,
    doi          = {10.5281/zenodo.3337331},
    url          = {<a href="https://doi.org/10.5281/zenodo.3337331">https://doi.org/10.5281/zenodo.3337331</a>}
}
"""
import pydicom
from pathlib import Path
from PIL import Image
import tensorflow_io as tfio
import tensorflow as tf
from skimage.exposure import rescale_intensity
import numpy as np


def load_dcm(
    fp: Path,
    on_error="lossy",
    scale="auto",
    dtype=tf.uint8,
    enhance=False,
    perc_lo=2,
    perc_hi=98,
    **kwargs
) -> np.array:
    """Converts a DICOM image located ad `fp` to a np.array, potentially enhancing it. 

    Args:
        fp (pathlib.Path): File path of the DICOM image
        on_error (str, optional): Establishes the behaviour in case an error occurs on opening the image or if the output type cannot accomodate all the possible input values. For example if the output dtype is set to np.uint8, but a dicom image stores a tf.uint16 dtype, "strict" throws an error. "skip" returns a 1-D empty tensor. "lossy" continues with the operation scaling the value via the scale attribute. Defaults to "lossy".
        scale (str, optional): Establishes how to scale the input values. "auto" will autoscale the input values according to the output dtype (e.g. uint8 -> [0, 255]). If the output is float, "auto" will scale to [0,1]. "preserve" keeps the values as they are, an input value greater than the maximum possible output will be clipped. Defaults to "auto".
        dtype ([type], optional): Output dtype. Defaults to np.uint8.
        enhance (bool, optional): Whether to apply contrast stretching by histogram clipping. Defaults to False.
        perc_lo (int, optional): Lower percentile for clipping. Ignored if "enhance" is False. Defaults to 2. 
        perc_hi (int, optional): Upper percentile for clipping. Ignored if "enhance" is False. Defaults to 98.

    Returns:
        np.array: The converted array.
    """
    ar = (
        tfio.image.decode_dicom_image(
            tf.io.read_file(str(fp)),
            on_error="lossy",
            scale="auto",
            dtype=np.uint8,
            **kwargs
        )
        .numpy()
        .squeeze()
    )
    if enhance:
        p2, p98 = np.percentile(ar, (perc_lo, perc_hi))
        return rescale_intensity(ar, in_range=(p2, p98))
    return ar


def ar_to_PIL(ar: np.array) -> Image:
    """Converts a np.array to a PIL Image.

    Args:
        ar (np.array): The array to convert. Usually produced by `load_dcm`.

    Returns:
        Image: a PIL.Image converted to monochrome.
    """
    return Image.fromarray(ar).convert("L")


def save_PIL(im: Image, path: str = None, **kwargs) -> None:
    """Saves a PIL.Image to a `path`.

    Args:
        im (Image): The PIL.Image to save
        path (str, required): The path to save to.

    Raises:
        ValueError: If "path" is not provided
    """
    if path is None:
        raise ValueError("Path must be specificed")
    im.save(fp=path, **kwargs)
    return
