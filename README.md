Learning Camera Miscalibration Detection
==============================

Tensorflow code for training and testing a deep convolutional neural network for detecting camera miscalibration.


## Setup Instructions

1. Clone this repository
```
git clone https://github.com/ethz-asl/camera_miscalib_detection.git
```
2. Install the following python dependencies:
```
pip3 install --user carnivalmirror opencv-python tensorflow-gpu==1.13.1 pandas numpy matplotlib
```
3. Download the KITTI dataset:
```
./utils/KITTI_download.sh
```

## Training the network

__NOTE:__ The processing of dataset happens inside the [dataset.py](dataset.py).

### Usage
```
train.py [-h] [-n_train_samples N_TRAIN_SAMPLES]
              [-n_valid_samples N_VALID_SAMPLES] [-batch_size BATCH_SIZE]
              [-buffer_size BUFFER_SIZE] [-epochs EPOCHS]
              [-model_path MODEL_PATH] [-log_path LOG_PATH]
              [-log_name LOG_NAME] [-checkpoints CHECKPOINTS] [-v V]
              [-njobs NJOBS]
              index train_selector valid_selector
```


## License
3-Clause BSD; see LICENSE

## References
The algorithm is based on the following papers. Please cite the appropriate papers when using parts of it in an academic publication.

1. A. Cramariuc, A. Petrov, R. Suri, M. Mittal, R. Siegwart, C. Cadena (2020). Learning Camera Miscalibration Detection. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), Paris, France
