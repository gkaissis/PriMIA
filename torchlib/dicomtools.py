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
from pathlib import Path
from PIL import Image
import tensorflow_io as tfio
import tensorflow as tf
from skimage.exposure import rescale_intensity
import numpy as np

from torchvision.datasets.folder import default_loader
from os.path import splitext
from typing import Dict, Union, Set, Callable

from .dataloader import single_channel_loader


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


def ar_to_PIL(ar: np.array, output_type="L") -> Image:
    """Converts a np.array to a PIL Image.

    Args:
        ar (np.array): The array to convert. Usually produced by `load_dcm`.
        output_type (str): PIL conversion [L=1 channel, R=3 channels]

    Returns:
        Image: a PIL.Image converted to monochrome.
    """
    return Image.fromarray(ar).convert(output_type)


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


class DicomLoader:
    """Takes a path for a dicom and returns a PIL Image.

    Args:
        out_channels (int): number of channels output image should contain

    Raises:
        KeyError: number of channels must be either 1 or 3
    """

    def __init__(self, out_channels: int = 3):
        self.mapping_channels_to_letters = {1: "L", 3: "RGB"}
        self.out_channels = out_channels

    def __call__(self, path: Path, **kwargs):
        """Return PIL image of path to dicom

        Args:
            fp (pathlib.Path): File path of the DICOM image
            kwargs: Passed to load_dcm

        Returns:
            Image: a PIL.Image converted to monochrome.

        """
        return ar_to_PIL(
            load_dcm(path, **kwargs),
            output_type=self.mapping_channels_to_letters[self.out_channels],
        )


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
