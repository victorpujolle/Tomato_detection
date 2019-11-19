# Mask R-CNN for object instance segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow.
The model generates bounding boxes and segmentation masks for each instance of an object in the image.
It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The repository provide on a simple implementation of the Mask R-CNN built on FPN
and ResNet101 and a Jupyter notebooks to train the network on a custom dataset using google colab.
It was created to detect tomatoes on an image, but it can be easily repurposed and trained.

# Get started
* Launch_inference.py is the main script
* Tomato.py provides classes needed to configure the model
* visualize.py provides functions needed to visualize inferences
* train_tomato.ipynb is a jupyter notebook usable to train the network using google colab


