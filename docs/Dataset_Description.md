# Dataset Description

The dataset was adapted from the [CoronaHack Chest X-Ray dataset](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset) and is used under the terms of the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode).

The following modifications were made to create a dataset with a (great) majority of frontal chest X-Rays of children and adolescents:
- All adult images (based on ossification status and/or degenerative changes to the spine) were removed. Example: ![](Removed/1-s2.0-S0929664620300449-gr2_lrg-a.jpg)
- All non frontal (i.e. not PA or AP) images were removed. Example:
![](Removed/4C4DEFD8-F55D-4588-AAD6-C59017F55966.jpeg)
- All CT-images were removed. Example:
![](Removed/1-s2.0-S0929664620300449-gr3_lrg-b.jpg)
- All non-grayscale images were removed. Example:
![](Removed/gr1_lrg-a.jpg)
- All images considered of non-diagnostic quality were removed. Example:
![](Removed/person1679_bacteria_4448.jpeg)

The removed images can be found in the `Removed` folder.

The training set consists of 5163 images, the test set of 624 images.

The original metadata file was modified to represent the new dataset. Labels were encoded as follows:

| Label | Description         |
|-------|---------------------|
| 0     | Normal              |
| 1     | Bacterial Pneumonia |
| 2     | Viral Pneumonia     |

The original metadata file is `Chest_xray_Corona_Metadata.csv` and the new one is `Labels.csv`. To make the new label file, run `Create_labels.ipynb`.

