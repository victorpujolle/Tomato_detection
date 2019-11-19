# Mask R-CNN for object instance segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow.
The model generates bounding boxes and segmentation masks for each instance of an object in the image.
It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The repository provide on a simple implementation of the Mask R-CNN built on FPN
and ResNet101 and a Jupyter notebooks to train the network on a custom dataset using google colab using pretrained weights
It was created to detect tomatoes on an image, but it can be easily repurposed and trained.

# Tomato folder
* ```Launch_inference.py``` is the main script
* ```Tomato.py``` provides classes needed to configure the model
* ```visualize.py``` provides functions needed to visualize inferences
* ```train_tomato.ipynb``` is a jupyter notebook usable to train the network using google colab

# Tomato dataset
I made a small dataset for tomato detection. pLease feel free to use it

[Tomato dataset](https://drive.google.com/drive/folders/1QUBwzUc8uyjCXemetmurzBxbxt4pWQZ1?usp=sharing)

# Training on your own dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46).
It covers the process starting from annotating images to training to using the results in a sample application.

To  train on your own dataset you need to extend two classes:

* ```Config``` This class contains the default configuration. Subclass it and modify the attributes you need to change.
* ```Dataset```  This class provides a consistent way to work with any dataset. It allows you to use new datasets for training without having to change the code of the model.
It also supports loading multiple datasets at the same time, which is useful if the objects you want to detect are not all available in one dataset.

See examples in `samples/shapes/train_shapes.ipynb`, `samples/coco/coco.py`, `samples/balloon/balloon.py`, and `samples/nucleus/nucleus.py`
from the [original repository](https://github.com/matterport/Mask_RCNN)

# Create your own dataset without having to write new code

You can search images on flickr, limiting the licence type to Commercial use & mods allowed. Between 75 and 100 images should be enough. You may need more images if you need very good accuracy
but in simple cases it will be enough because we use transfert learning, meaning we don't train the model from scratch but start with a weight file that’s been trained on the COCO dataset.

Then divide them into a training set and a validation set, named ```train``` and ```val```.

There is a lot of tools to annotate images. The code here is made for [Via (VGG image annotator)](http://www.robots.ox.ac.uk/~vgg/software/via/).
It’s a single HTML file that you download and open in a browser.

Then you can annotate your images, using only the polygon annotation tool (you can use other tools but you'll have to extend the ```Dataset``` class)
Save the annotation using the JSON file,  each mask is a set of polygon points.
You should create 2 JSON files, for the training and the validation set. The name of the file should be ```via_region_data.json```.

Then, upload your dataset on your google drive and go on google colab.

# Get started
Open the ```train_tomato.ipynb``` in google colab and follow the instructions

