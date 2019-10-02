"""
Mask R-CNN
Configurations and data loading code for Fashion dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 fashion.py train --dataset=/path/to/fashion/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 fashion.py train --dataset=/path/to/fashion/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 fashion.py train --dataset=/path/to/fashion/dataset --weights=imagenet
    
    # Detect fashion items in an image
    python3 fashion.py detect --weights=/path/to/weights/file.h5 --images=<path to folder>
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# get this file's directory
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.join(FILE_DIR, "..")
sys.path.append(SRC_DIR)
from utils import get_labels, get_labels_colors, apply_masks_on_image, draw_bbox_and_labels_on_image

# Import Mask RCNN
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import model as modellib, utils

# Path to trained weights file
MASKRCNN_DIR = os.path.join(SRC_DIR, "Mask_RCNN")
COCO_WEIGHTS_PATH = os.path.join(MASKRCNN_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --models
DEFAULT_MODELS_DIR = os.path.join(SRC_DIR, "../models")

# annotation files
DATA_DIR = os.path.abspath("data")
LABELS_JSON_FILENAME = "label_descriptions.json"
ANNOTATIONS_CSV_FILENAME = "train.csv"

# Detection/Segmentation output directory
OUT_DIR = os.path.abspath("results")

############################################################
#  Configurations
############################################################

class FashionConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fashion"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 46 + 1  # Background + fashion

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000


############################################################
#  Dataset
############################################################
    
    
class FashionDataset(utils.Dataset):

    def __init__(self, dataset_dir, df, label_names):
        super().__init__(self)
        self.label_names = label_names
    
        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("fashion", i+1, name)
        
        # Add images 
        for i, row in df.iterrows():
            self.add_image("fashion", 
                           image_id=row.name, 
                           path=os.path.join(dataset_dir,'train',row.name), 
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])
    
    def image_reference(self, image_id):
        ''' Returns the image path and its labels for debugging purposes'''
        info = self.image_info[image_id]
        return info['path'], [self.label_names[int(x)] for x in info['labels']]
    
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
                
        mask = np.zeros((info["height"], info["width"], len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            
        return mask, np.array(labels)


def load_datasets(data_dir, lables_json_filename, annotations_csv_filename):
    
    # Get fashion labels from json file
    label_names = get_labels(data_dir, lables_json_filename)
    
    # read annotations CSV file into a pandas dataframe
    annotations_csv_path = os.path.join(data_dir, annotations_csv_filename)
    segment_df = pd.read_csv(annotations_csv_path)
    
    # discard the segments that contains attributes and only keep the category IDs
    segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]
    
    # aggregate segments to only keep one row by image in the dataframe
    image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
    size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
    image_df = image_df.join(size_df, on='ImageId')
    
    # split train / val datasets
    df_train, df_val = train_test_split(image_df, test_size=0.2, random_state=42)
    
    # load training dataset.
    dataset_train = FashionDataset(data_dir, df_train, label_names)
    dataset_train.prepare()

    # load validation dataset
    dataset_val = FashionDataset(data_dir, df_val, label_names)
    dataset_val.prepare()
    
    return dataset_train, dataset_val
    

def train(model):

    """Train the model."""
    
    # load train and valid datasets
    dataset_train, dataset_val = load_datasets(args.dataset,
                                                LABELS_JSON_FILENAME,
                                                ANNOTATIONS_CSV_FILENAME)
    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


def detect(model, images_path):

    ''' Perform object detection and segmentation on images'''
    
    # Output directory for images with Mask R-CNN annotations trained on fashion
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    
    # Load images file names from the images folder
    file_names = next(os.walk(images_path))[2]
    
    # Get fashion labels from json file
    CLASS_NAMES = get_labels(DATA_DIR, LABELS_JSON_FILENAME)
    CLASS_NAMES.insert(0, 'BG')

    # generate mask and bbox colors to annotate each label
    COLORS = get_labels_colors(CLASS_NAMES)

    # loop through images
    for i, file_name in enumerate(file_names):
        
        # read image with opencv and convert to RGB for Mask R-CNN input
        image_brg = cv2.imread(os.path.join(images_path, file_name))
        image = cv2.cvtColor(image_brg, cv2.COLOR_BGR2RGB)
        
        # Run Mask R-CNN detection/segmentation
        results = model.detect([image])[0]
        
        # apply mask to each object in the image
        image = apply_masks_on_image(image, results, COLORS)
            
        # draw bounding boxes, class labels, and score of each detection on the image
        image = draw_bbox_and_labels_on_image(image, results, COLORS, CLASS_NAMES)
        
        # save output
        cv2.imwrite(os.path.join(OUT_DIR, file_name), image)
        print('{}/{}: {}'.format(i+1, len(file_names), file_name))


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect fashion items.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/fashion/dataset/",
                        help='Directory of the Fashion dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--models', required=False,
                        default=DEFAULT_MODELS_DIR,
                        metavar="/path/to/models/",
                        help='Logs and checkpoints directory (default=models/)')
    parser.add_argument('--images', required=False,
                        metavar="path to images folder",
                        help='Images to detect fashion items on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.images, "Provide --images to detect fashion items"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Models: ", args.models)

    # Configurations
    if args.command == "train":
        config = FashionConfig()
    else:
        class InferenceConfig(FashionConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            #DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90% confidence
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.models)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.models)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, images_path=args.images)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
