# PointNet.pytorch
This repo is implementation application of PointNEt on RoofN3D dataset.

For Pointnet details, please look here:
1. PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet.py`.
2. The code has been taken from here: https://github.com/fxia22/pointnet.pytorch

# INSTRUCTIONS
## Classification
### training
1. download the data from https://drive.google.com/open?id=1JRKiO8_SOjoz6lqUK819-9cpO0Ag2Yr1
2. Unzip it into a folder
3. Change the data folder in the train_classification.py
4. run train_classification.py
### testing
1. run the test_cls.py with appropriate data folder.

# Acknowledgement
The RoofN3D training data (Wichmann et al., 2018) was provided by the chair Methods of Geoinformation Science of Technische Universität Berlin and is available at https://roofn3d.gis.tu-berlin.de.

# Citations
[1] Wichmann, A., Agoub, A., Kada, M., 2018. RoofN3D: Deep Learning Training Data for 3D Building Reconstruction. In: The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLII-2, pp. 1191-1198.

[2] Qi Charles, R & Su, Hao & Kaichun, Mo & Guibas, Leonidas. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. 77-85. 10.1109/CVPR.2017.16. 


