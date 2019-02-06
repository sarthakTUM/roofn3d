# Roof Classification, Segmentation, and Damage Completion using 3D Point Clouds

This repository contains the project for the course: https://dvl.in.tum.de/teaching/adl4cv-ws18/

## About
We design a Deep Learning framework that directly consumes unordered point sets as inputs, representing the Roof of a building. 
A Point Cloud is represented as a set of 3D points {Pi| i = 1, ..., n}, where each point Pi is a vector of its (x, y, z)
coordinate. We perform the following tasks:
1. Classification: Clasifying the input point cloud into 3 categories: Saddleback Roof, Pyramid Roof, and Two-Sided Hip Roof
2. Segmentation: Classifiying each point into semantic sub-categories.
3. Damaging: Manually simulating damages of various shapes in the roofs.
4. Completion: Completing the damaged point cloud
5. Multi-Task Learning: Perform Classification, and Segmentation in a multi-task setting for original, and perturbed dataset.

## Approach
1. PointNet [2] is used for Classification, Segmentation and Multi-Task Learning on original as well as perturbed dataset
2. Point Completion Network [3] is used for completing the damaged roofs.
3. The damage has been manually simulated on the roof point clouds.
4. Data has been provided by [1].

## Usage
### Demo
1. Clone the repository: `git clone https://github.com/sarthakTUM/roofn3d.git`
2. Install the requirements: `pip install requirements.txt`. It is recommended to perform this step in a separate virtual environment.
3. For classification, segmentation, and multi-task demo: `cd cls_seg_mt`. 
4. Run `python demo.py`. Different examples can be seen by changing the `--idx`parameter. 
For example, `python demo.py --idx=15`. The `--idx` parameter can be a maximum of 23.
5. For Roof Completion, go to `cd completion` from the roof directory.
6. Run `python demo.py`. Different example can be seen by changing the `--input_path` parameter. For example: `python demo.py --input_path=demo_data/saddleback_roof.pcd`, or `python demo.py --input_path=demo_data/twosidedhip_roof.pcd`.
7. Demo data for both the tasks have been added to demo:data directories. 

The models are cloned along with the repository. If there are any difficulties in cloning the models, please download the models
from:
1. https://drive.google.com/open?id=1C8X4O9SnNmvmJbzpqx_2YJR7sYTxZ3Tm for Classification, segmentation, and Multi-Task networks. Place the models in the `roofn3d/cls_seg_mt/models` directory.
2. https://drive.google.com/open?id=15r54fXLjZcFuL2ok35M1_hXxw_Fn7LJu for Completion. Place the model in `completion/log` directory. 

### Training
#### Classification, Segmentation, and Multi-Task
##### Non-Damaged data
download the data from https://drive.google.com/open?id=1JRKiO8_SOjoz6lqUK819-9cpO0Ag2Yr1
##### Damaged data
download the data from <insert link here>
1. Unzip it into a folder
2. go to `cd cls_seg_mt`. Run `python train_classification.py --input_path=path_to_data_from_step2 --outf=models/cls`.
You must change the `--input-path`to path of data dobtained from step 2. The  `outf` argument corresponds to output drectory for the trained models.
3. for segmentation, run `python train_segmentation.py --input_path=path_to_data_from_step2 --outf=models/seg`. 
You must change the `--input-path`to path of data dobtained from step 2. The  `outf` argument corresponds to output drectory for the trained models.
4. For Multi-Task Learning, run `python train_multitask.py --input_path=path_to_data_from_step2 --outf=models/mt`. 
You must change the `--input-path`to path of data dobtained from step 2. The  `outf` argument corresponds to output drectory for the trained models.

**NOTE**: The first run might take some time to load all the point-clouds in the memory and save them for faster access in the next run.
It is recommended to spare atleast 10GB of RAM for data loading.

### Testing
1. go to `cd cls_seg_mt`, and run `python test_cls.py --input_path=path_to_data --model=model_to_test.pth`
2. go to `cd cls_seg_mt`, and run `python test_seg.py --input_path=path_to_data --model=model_to_test.pth`
3. go to `cd cls_seg_mt`, and run `python test_multitask.py --input_path=path_to_data --model=model_to_test.pth`

# Acknowledgement
The RoofN3D training data (Wichmann et al., 2018) was provided by the chair Methods of Geoinformation Science of Technische Universität Berlin and is available at https://roofn3d.gis.tu-berlin.de.

# Citations
[1] Wichmann, A., Agoub, A., Kada, M., 2018. RoofN3D: Deep Learning Training Data for 3D Building Reconstruction. In: The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLII-2, pp. 1191-1198.

[2] Qi Charles, R & Su, Hao & Kaichun, Mo & Guibas, Leonidas. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. 77-85. 10.1109/CVPR.2017.16. 

[3] Yuan, Wentao & Khot, Tejas & Held, David & Mertz, Christoph & Hebert, Martial. (2018). PCN: Point Completion Network. 728-737. 10.1109/3DV.2018.00088. 
