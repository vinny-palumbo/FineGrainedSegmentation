"""
Mask R-CNN
Configurations and inference code for Fashion dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Modified by Vincent Palumbo

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    
    # Detect fashion items in an image
    python3 detect.py --weights=/path/to/weights/file.h5 --images=<path to folder>
"""

import os
import sys
import cv2

# get this file's directory and add src/ to sys.path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.join(FILE_DIR, "..")
sys.path.append(SRC_DIR)
import utils as utils_fashion

# Import Mask RCNN
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --models
DEFAULT_MODELS_DIR = os.path.join(SRC_DIR, "../models")

# necessary annotation files in the data
DATA_DIR = os.path.abspath("data")
LABELS_JSON_FILENAME = "label_descriptions.json"

# Detection/Segmentation output directory
OUT_DIR = os.path.abspath("results")

############################################################
#  Configurations
############################################################

class InferenceConfig(Config):

    '''Configuration for inference on the fashion dataset.
    Derives from the base Config class and overrides some values.
    '''

    # Give the configuration a recognizable name
    NAME = "fashion"
    
    # Number of classes (including background)
    NUM_CLASSES = 46 + 1  # Background + fashion

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MIN_CONFIDENCE = 0.9


def detect(model, images_path):

    ''' Perform object detection and segmentation on images'''
    
    # Output directory for images with Mask R-CNN annotations trained on fashion
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    
    # Load images file names from the images folder
    file_names = next(os.walk(images_path))[2]
    
    # Get fashion labels from json file
    CLASS_NAMES = utils_fashion.get_labels(DATA_DIR, LABELS_JSON_FILENAME)
    CLASS_NAMES.insert(0, 'BG')

    # generate mask and bbox colors to annotate each label
    COLORS = utils_fashion.get_labels_colors(CLASS_NAMES)

    # loop through images
    for i, file_name in enumerate(file_names):
        
        # read image with opencv and convert to RGB for Mask R-CNN input
        image_brg = cv2.imread(os.path.join(images_path, file_name))
        image = cv2.cvtColor(image_brg, cv2.COLOR_BGR2RGB)
        
        # Run Mask R-CNN detection/segmentation
        results = model.detect([image])[0]
        
        # apply mask to each object in the image
        image = utils_fashion.apply_masks_on_image(image, results, COLORS)
            
        # draw bounding boxes, class labels, and score of each detection on the image
        image = utils_fashion.draw_bbox_and_labels_on_image(image, results, COLORS, CLASS_NAMES)
        
        # save output
        cv2.imwrite(os.path.join(OUT_DIR, file_name), image)
        print('{}/{}: {}'.format(i+1, len(file_names), file_name))


############################################################
#  Detect
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Detect and segment fashion items in images.')
    parser.add_argument('--images', required=True,
                        metavar="path to images folder",
                        help='Images to detect fashion items on')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--models', required=False,
                        default=DEFAULT_MODELS_DIR,
                        metavar="/path/to/models/",
                        help='Logs and checkpoints directory (default=models/)')
    args = parser.parse_args()

    # Validate arguments
    assert args.images, "Provide --images to detect fashion items"
    assert args.weights, "Provide --weights to detect fashion items"
    
    print("Images: ", args.images)
    print("Weights: ", args.weights)
    print("Models: ", args.models)

    # Configurations
    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.models)
    
    # Load weights
    model.load_weights(args.weights, by_name=True)

    # detect and segment objects in images
    detect(model, images_path=args.images)

