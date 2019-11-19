import os
import sys
import random

from Tomato import TomatoConfig, TomatoDataset

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

# Path to the dataset
DATASET_PATH = os.path.join(ROOT_DIR, 'Real_dataset')

# Configuration file for training and inference
config = TomatoConfig()
config.display()

# Validation dataset
dataset_val = TomatoDataset()
dataset_val.load_tomato(DATASET_PATH, "val")
dataset_val.prepare()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# Get path to saved weights
TOMATO_MODEL_PATH = os.path.join(ROOT_DIR, 'logs/mask_rcnn_tomato.h5')

# Load trained weights
print("Loading weights from ", TOMATO_MODEL_PATH)
model.load_weights(TOMATO_MODEL_PATH, by_name=True)


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

#visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
#visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], ax=get_ax())