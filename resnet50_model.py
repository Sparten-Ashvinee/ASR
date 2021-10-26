import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


def resnet(device):
    model = models.resnet50(pretrained=True)
    model.conv1=nn.Conv2d(1, model.conv1.out_channels,
                          kernel_size=model.conv1.kernel_size[0],
                          stride=model.conv1.stride[0],
                          padding=model.conv1.padding[0])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)

    cnn = model
    summary(cnn, (1, 64, 44))

    for name, param in cnn.named_parameters():
      # if the param is from a linear and is a bias
      if "linear" in name and "bias" in name:
        param.register_hook(lambda grad: torch.zeros(grad.shape))

    return cnn