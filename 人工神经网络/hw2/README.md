# MLP and CNN on CIFAR-10

## MLP Implementation

No modification of framework code.

## CNN Implementation

Added `vgg_model` parameter in `main.py`. When executing with this parameter, a quasi-VGG model is used, yielding a better performance than legacy models. Other parts remain the same as original framework.

