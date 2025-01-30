![mit-logo](https://csailprettycommittee.mit.edu/sites/default/files/images/MIT_logo.png)

![alt_tag](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)

# Pytorch Tutorial for Fine Tuning/Transfer Learning a Resnet for Image Classification

If you want to do image classification by fine tuning a pretrained mdoel, this is a tutorial will help you out. It shows how to perform fine tuning or transfer learning in PyTorch with your own data. It is based on a bunch of of official pytorch tutorials/examples. I felt that it was not exactly super trivial to perform in PyTorch, and so I thought I'd release my code as a tutorial which I wrote originally for my research. 

Highly encourage you to run this on a new data set (read main_fine_tuning.py to know which format to store your data in), but for a sample dataset to start with, you can download a simple 2 class dataset from here - https://download.pytorch.org/tutorial/hymenoptera_data.zip

All Torch and PyTorch specific details have been explained in detail in the file main_fine_tuning.py

Hope this tutorial helps you out! :)

Credits - This tutorial is built on top of mainly on 2 Pytorch tutorials - http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html and https://github.com/pytorch/examples/tree/master/imagenet.
