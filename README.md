# Anno-Mage: A Semi Automatic Image Annotation Tool

![alt text](https://raw.githubusercontent.com/virajmavani/semi-auto-image-annotation-tool/master/demo.gif)

Semi Automatic Image Annotation Toolbox with tensorflow and keras object detection models.

## Installation

1) Clone this repository.

2) In the repository, execute `pip install -r requirements.txt`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
   Also, make sure Keras 2.1.3 or higher and OpenCV 3.x is installed.

3) a) For Keras model - Download the [pretrained weights](https://github.com/fizyr/keras-retinanet/releases/download/0.3.1/resnet50_coco_best_v2.1.0.h5) and save it in /snapshots/keras.

   b) For tensorflow model get the desired model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and extract it in /sanpshots/tensorfow
   
   c) You can even save custom pre trained model in the respective directory.
  
   

### Dependencies

1) Tensorflow >= 1.7.0

2) OpenCV = 3.x

3) Keras >= 2.1.3

For, Python >= 3.5

### Instructions

1) Select the COCO object classes for which you need suggestions from the drop-down menu and add them. Or simply click on ```Add all classes``` .

2) Select the desired model and click on ```Add model```.

3) Click on ```detect``` button.

4) When annotating manually, select the object class from the List and while keep it selected, select the BBox.

5) The final annotations can be found in the file `annotations.csv` in ./annotations/ . Also a xml file will saved.

### Usage

For MSCOCO dataset
```
python main.py
```
For any other dataset-

First change the labels in config.py (for keras model) or in tf_config.py( for tensorflow model).
Then run:
```
python main.py
```

#### Tested on:
1. Windows 10

2. Linux 16.04

3. macOS High Sierra

### Join the developers channel for contributions

Slack: https://join.slack.com/t/annomage/shared_invite/zt-dh4ca9du-4VOcwUMCSNA6lmyG~tNUPg

### Acknowledgments

1) [Meditab Software Inc.](https://www.meditab.com/)

2) [Keras implementation of RetinaNet object detection](https://github.com/fizyr/keras-retinanet)

3) [Computer Vision Group](https://cvgldce.github.io/), L.D. College of Engineering
