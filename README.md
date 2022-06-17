# OptiForML2022

Project  of the Optimization for Machine Learning course given at the EPFL 2022.

The goal of this project is to study adaptative meta optimizer as Atmo on image classification task (MNIST, cifar10) using deep network architecture (resnet18). We also study second-order optimizers and combinations of second order and first order optimizers.

## Team members

- Arthur Nussbaumer
- Emilien Seiler
- Louis Poulain--Auzéau

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

## Run

The notebooks are adapted to be ran on Google Colab. The third cell can be skipped if you run on your local machine.
To run the training use the notebook `train_optimizer.ipynb` choose optimizer, scheduler and number of epochs in the fourth cell.  
To plot the training log use the notebook `result_visualization.ipynb`.

## Structure

This is the structure of the repository:

- `fig`: figure
- `optimizer`: contains different optimizer for training
  - `AdaHessian.py`: adahessian second order optimizer implementation
  - `Atmo.py` : atmo meta optimizer implementation
  - `Padam_SGD.py`: padam sgd mix
  - `AdaSGD.py`: adahessian sgd mix
- `output`: contains output of training
  - `pretrained_model`: model save after training
  - `training_history`: logs for each training. Each log is a dictionnary that contains the train/test losses and accuracies, the time per epoch, the name of the optimizer and of the scheduler
  - `archive`: logs of training that we didn't include in the paper. In particular, some concern the MNIST dataset, which was used as a test
- `dataset.py`: contain pytorch dataset of image (MNIST or cifar10)
- `model.py`: contain model use for train (resnet18)
- `path.py`: path management
- `result_visualization.ipynb`: notebook to visualize training metric according to different optimizer
- `train_optimizer.ipynb`: notebook use to train different optimizer can be use on google collab (import the git on your drive)

## References

See [references](references.md).
