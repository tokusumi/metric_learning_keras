# Siamese Network with L2softmaxloss
Implemented with Keras

## Overview
This repository includes a comparison between L2-softmaxloss vs softmaxloss for metric learning.
I constructed Siamese network to calculate pointwise similarity between two images.

See [notebook](https://github.com/tokusumi/metric_learning_keras/blob/master/L2SoftmaxLoss/compare_L2SoftmaxLoss.ipynb)

## Dataset
MNIST handwritten digit image dataset: http://yann.lecun.com/exdb/mnist/

## Results
![PR curve](https://github.com/tokusumi/metric_learning_keras/blob/master/L2SoftmaxLoss/PR_curve_l2.png)

## Refs

### L2-SoftmaxLoss (L2-constrained Softmax Loss)

> Y. Wen, K. Zhang, Z. Li, and Y. Qiao. A discrimina- tive feature learning approach for deep face recognition. In European Conference on Computer Vision, pages 499–515. Springer, 2016.

