# Dataset Description

The dataset in this repository is being re-used under the license terms from [the CoronaHack Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset) and modified as indicated in `Dataset_Description.md`.

Original citation for the majority of the images: https://data.mendeley.com/datasets/rscbjbr9sj/2

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

The following modifications were made to create a dataset with a (great) majority of frontal chest X-Rays of children and adolescents:

- All adult images (based on ossification status and/or degenerative changes to the spine) were removed. Example: ![](images/removed_images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg)

- All non frontal (i.e. not PA or AP) images were removed. Example:
![](images/removed_images/4C4DEFD8-F55D-4588-AAD6-C59017F55966.jpeg)

- All CT-images were removed. Example:
![](images/removed_images/1-s2.0-S0929664620300449-gr3_lrg-b.jpg)

- All non-grayscale images were removed. Example:
![](images/removed_images/gr1_lrg-a.jpg)

- All images considered of non-diagnostic quality were removed. Example:
![](images/removed_images/person1679_bacteria_4448.jpeg)

The removed images can be found in the `Removed` folder.

The training set consists of 5163 images, the test set of 624 images.

The original metadata file was modified to represent the new dataset. Labels were encoded as follows:

| Label | Description         |
|-------|---------------------|
| 0     | Normal              |
| 1     | Bacterial Pneumonia |
| 2     | Viral Pneumonia     |

The original metadata file is `data/Chest_xray_Corona_Metadata.csv` and the new one is `data/Labels.csv`.


