### PLR Miscalibration Detection

#### Running Tensorflow through a Docker container

Note that you need to have `nvidia-docker` installed.
```
docker run -v "$(pwd)":/files -it --rm --runtime=nvidia tensorflow/tensorflow:1.12.3-gpu-py3 bash
```