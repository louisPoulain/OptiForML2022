# OptiForML2022

Project  of the Optimization for Machine Learning course given at the EPFL 2022.

The goal of this project is to study adaptative meta optimizer as Atmo on image classification task (MNIST and cifar10) using deep network architecture (resnet18)

## Team members

- Arthur
- Emilien Seiler
- Louis

## Installing

To run the code of this project, you need to install the libraries listed in
the `requirements.txt` file. You can perform the installation using this
command:
```
pip3 install -r requirements.txt
```
Dependencies:
- matplotlib
- numpy
- pickle
- torch
- torchvision
- tqdm

## Structure

This is the structure of the repository:

-`fig`: 
  -`Cifar_training_metric.png`: figure with training metric on cifar dataset according to each optimizer
  -`MNIST_training_metric.png`: figure with training metric on MNIST dataset according to each optimizer
- `optimizer`: contains different optimizer for training
  - `AdaHessian.py`: adahessian second order optimizer implementation
  - `Atmo.py` : atmo meta optimizer implementation
- `output`: contains output of training
  - `pretrained_model`: model save after training
  - `training_history`: loss, acc, ... of training
- `dataset.py`: contain pytorch dataset of image (MNIST or cifar10)
- `model.py`: contain model use for train (resnet18)
- `path.py`: path management
- `result_visualization.ipynb`: notebook to visualize training metric according to different optimizer
- `train_optimizer.ipynb`: notebook use to train different optimizer can be use on google collab (import the git on your drive)

## References

See [references](references.md).
