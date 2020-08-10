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


def load_dcm(f: Path) -> np.array:
    d = pydicom.read_file(f)
    im_ar = d.pixel_array
    slope = float(d.RescaleSlope) if hasattr(d, "RescaleSlope") else 1
    flip = -1 if d.PresentationLUTShape == "INVERSE" else 1
    mult = slope * flip
    intercept = float(d.RescaleIntercept) if hasattr(d, "RescaleIntercept") else 0
    rescaled = (im_ar * mult) + intercept
    normed = (rescaled - rescaled.min()) / (rescaled.max() - rescaled.min())
    return np.array(Image.fromarray(normed * 255).resize((224, 224)).convert("L"))

