import torch
import os
from torchvision import datasets
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, full=False, tiny=False, cifar=False, test=False):
        """images dataset initialization

        Args:
            full (bool, optional): Use the full set. Defaults to False.
            tiny (bool, optional): Use a very small set for quick checks.
            Defaults to False.
            cifar (bool, optional): Use the CIFAR data-set and not MNIST .
            Defaults to False.
            test (bool, optional): load test data and not train.
            Defaults to False.
        """
        self.full = full
        self.tiny = tiny
        self.cifar = cifar
        self.test = test

        # load data
        self.load_data()

    def __getitem__(self, index):
        return self.x[index, ...], self.y[index]

    def __len__(self):
        return self.y.shape[0]

    def load_data(self):
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

        if self.cifar:
            print('* Using CIFAR')
            if not(self.test):
                cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/',
                                                   train=True,
                                                   download=True)
                train_input = torch.from_numpy(cifar_train_set.data)
                train_input = train_input.transpose(3, 1).transpose(2, 3)
                train_input = train_input.float()
                train_target = torch.tensor(cifar_train_set.targets,
                                            dtype=torch.int64)

            else:
                cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/',
                                                  train=False,
                                                  download=True)
                test_input = torch.from_numpy(cifar_test_set.data).float()
                test_input = test_input.transpose(3, 1).transpose(2, 3).float()
                test_target = torch.tensor(cifar_test_set.targets,
                                           dtype=torch.int64)

        else:
            print('* Using MNIST')
            if not(self.test):
                mnist_train_set = datasets.MNIST(data_dir + '/mnist/',
                                                 train=True, download=True)
                train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
                train_target = mnist_train_set.targets
            else:
                mnist_test_set = datasets.MNIST(data_dir + '/mnist/',
                                                train=False, download=True)
                test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
                test_target = mnist_test_set.targets
        if self.full:
            if self.tiny:
                raise ValueError('Cannot have both --full and --tiny')
        else:
            if self.tiny:
                print('** Reduce the data-set to the tiny setup')
                if not(self.test):
                    train_input = train_input.narrow(0, 0, 500)
                    train_target = train_target.narrow(0, 0, 500)
                else:
                    test_input = test_input.narrow(0, 0, 100)
                    test_target = test_target.narrow(0, 0, 100)
            else:
                print('** Reduce the data-set (use --full for the full thing)')
                if not(self.test):
                    train_input = train_input.narrow(0, 0, 1000)
                    train_target = train_target.narrow(0, 0, 1000)
                else:
                    test_input = test_input.narrow(0, 0, 1000)
                    test_target = test_target.narrow(0, 0, 1000)
        if not(self.test):
            print(f'** Use {train_input.size(0)} train samples')
            self.x = train_input
            self.y = train_target
        else:
            print(f'** Use {test_input.size(0)} test samples')
            self.x = test_input
            self.y = test_target
