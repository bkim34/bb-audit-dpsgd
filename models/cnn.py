"""Convolutional Neural Networks from https://github.com/OsvaldFrisk/dp-not-all-noise-is-equal/blob/master/src/networks.py
Paper: https://arxiv.org/pdf/2110.06255"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_shape=None, out_dim=10, dropout_rate=0):
        super().__init__()
        if in_shape[1] == 1:
            # MNIST
            self.net = SmallNetwork(out_dim=out_dim, dropout_rate=dropout_rate)
        elif in_shape[1] == 3:
            # CIFAR-10
            self.net = BigNetwork(out_dim=out_dim, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.net(x)

class BigNetwork(nn.Module):
    """Network used in the experiments on CIFAR-10"""

    def __init__(self, act_func=nn.Tanh, input_channels: int = 3, out_dim = 10, dropout_rate=0):
        super(BigNetwork, self).__init__()
        self.in_channels: int = input_channels

        # Variables to keep track of taken steps and samples in the model
        self.n_samples: int = 0
        self.n_steps: int = 0

        # Feature Layers
        feature_layer_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
        feature_layers = []

        c = self.in_channels
        for v in feature_layer_config:
            if v == 'M':
                feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                feature_layers += [conv2d, act_func()]
                c = v
        self.features = nn.Sequential(*feature_layers)

        # Classifier Layers
        num_hidden: int = 128
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(c * 4 * 4, num_hidden), act_func(), nn.Linear(num_hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(self.dropout(x))
        return x

class SmallNetwork(nn.Module):
    """Network used in the experiments on MNIST"""

    def __init__(self, act_func=torch.tanh, out_dim=10, dropout_rate=0.0) -> None:
        super(SmallNetwork, self).__init__()

        # Variables to keep track of taken steps and samples in the model
        self.n_samples: int = 0
        self.n_steps: int = 0

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, out_dim)

        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_func(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.act_func(F.max_pool2d(self.conv2(x), (2, 2)))
        x = x.view(-1, 512)
        x = self.act_func(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x