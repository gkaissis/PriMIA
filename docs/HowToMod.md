# Modifying PriMIA to other use-cases

PriMIA is designed as a generic library. If you want to use our techniques to train on different data (e.g. a different number of classes) or on another task altogether (segmentation instead of classification), you can do so with a few simple modifications.

## If your task is still a classification task, but you have a different number of classes

## If your task is a regression task which only requires switching out the loss function and final activation function

## If you want to switch out the network architecture only, but for the same task

## If your task is a non-classification task and requires a different network architecture