# Deep Learning Project using PyTorch

Overview

In this project, I developed an image classifier using PyTorch. The code was compiled into a command line application.
The classifier was tested on a dataset containing different types of flowers. However, you could apply the same workflow on your a dataset of your choice.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [pytorch](https://pytorch.org/)

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github. I was using a training set of 102 different types of flowers, where there ~20 images per flower to train on.  Then I used the trained classifier to see if I can predict the type for new images of the flowers.

## Project Files

- train.py: contains the pytorch code needed to load and train the deep neural network.
- predict.py: contains the code for testing the classifier to new test cases.

The trained model was saved into the checkpoint.pth file. You can load this model in predict.py and test it on new images or you can build your model from scratch using the train.py file.