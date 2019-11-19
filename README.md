# Mask R-CNN for object instance segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow.
The model generates bounding boxes and segmentation masks for each instance of an object in the image.
It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The repository provide on a simple implementation of the Mask R-CNN built on FPN
and ResNet101 and a Jupyter notebooks to train the network on a custom dataset using google colab using pretrained weights
It was created to detect tomatoes on an image, but it can be easily repurposed and trained.

# Tomato folder
* ```Launch_inference``` is the main script
* ```Tomato.py provides``` classes needed to configure the model
* ```visualize``` provides functions needed to visualize inferences
* ```train_tomato``` is a jupyter notebook usable to train the network using google colab

# Tomato dataset
I made a small dataset for tomato detection. pLease feel free to use it
[Tomato dataset](https://drive.google.com/drive/folders/1QUBwzUc8uyjCXemetmurzBxbxt4pWQZ1?usp=sharing)

# Train on your own dataset
```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset.
It allows you to use new datasets for training without having to change
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not
all available in one dataset.
