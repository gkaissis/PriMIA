"""Reference implementation of loading a DICOM file from a pathlib.Path.
Notes:
- Some images in our test set don't have a RescaleSlope, thus we assume the slope is 1, (similar for RescaleIntercept)
- The PresentationLUTShape check is there because some x-ray machines save the image as inverted
- This implementation does a resize (not a center crop like our architecture)
-
"""
import pydicom
from pathlib import Path
from PIL import Image


def load_dcm(filepath: Path) -> np.array:
    """Load a DICOM located at `filepath` to a numpy array.

    Args:
        f (Path): Path of the image to load.

    Returns:
        np.array: ndarray of shape a x b x 1 channel.
    """
    d = pydicom.read_file(filepath)
    im_ar = d.pixel_array
    slope = float(d.RescaleSlope) if hasattr(d, "RescaleSlope") else 1
    flip = (
        -1
        if (hasattr(d, "PresentationLUTShape") and d.PresentationLUTShape == "INVERSE")
        else 1
    )
    mult = slope * flip
    intercept = float(d.RescaleIntercept) if hasattr(d, "RescaleIntercept") else 0
    rescaled = (im_ar * mult) + intercept
    normed = (rescaled - rescaled.min()) / (rescaled.max() - rescaled.min())
    return np.array(Image.fromarray(normed * 255).resize((224, 224)).convert("L"))

