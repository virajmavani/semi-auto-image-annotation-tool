# Semi Automatic Image Annotation Toolbox

![alt text](https://raw.githubusercontent.com/virajmavani/semi-auto-image-annotation-tool/master/data/demo.png)

Semi Automatic Image Annotation Toolbox with RetinaNet as the suggesting algorithm. The toolbox suggests 80 class objects from the MS COCO dataset using a pretrained RetinaNet model.

## Installation

1) Clone this repository.

2) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
   Also, make sure Keras 2.1.3 or higher is installed.

3) Download the [pretrained weights](https://github.com/fizyr/keras-retinanet/releases/download/0.2/resnet50_coco_best_v2.0.2.h5) and save it in /snapshots.

### Dependencies

1) Tensorflow >= 1.7.0

2) OpenCV >= 3.4.0

3) Keras >= 2.1.3

For, Python >= 3.5

### Usage
```
python main.py
```
