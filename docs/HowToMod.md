# Contributing to or modifying PriMIA

PriMIA is designed as a generic library. If you want to use our framework to train on different data (e.g. a different number of classes) or on another task altogether (segmentation instead of classification), here are some pointers to get you started.

## General considerations
In general, if you are interested in adapting PriMIA to a different use case, please get in touch with us and join the OpenMined team! We are an open, diverse and welcoming community and appreciate anyone who wants to get involved!

## If you want to switch out the network architecture
... consider using one of the networks provided, which are suitable for distributed systems. Large networks (like ResNet50) are compatible, but might be impractical to use due to the I/O overhead. That said, to include a new network, modify `torchlib/models.py` and adapt the training loop in `train.py` using standard PyTorch practices.

## If you want to modify the data used
... you need to change the dataloader used, which can be found in `torchlib/dataloader.py` for a dataloader suitable to your task. You will probably also need to modify the loss function, which can be found in `torchlib/utils.py`

## If you want to use similar data with minor modifications to the training task
... you can probably get by with minor modifications to the training loop in `train.py` and potentially the dataloader and loss function as discussed above. 

## If you want to port PriMIA to PySyft 0.3.0
... get in touch with us! We are happy to support you.

## If you want to port PriMIA to TensorFlow or JAX
... _definitely_ get in touch with us! We'd be very excited!

## If you find a bug or want to suggest a feature
... open a GitHub issue and get in touch with us on the OpenMined Slack! Hint: we have a PriMIA logo next to our names :)

## If you like our work
... spread the word! Not just about us, but about privacy-preserving machine learning in general!  
