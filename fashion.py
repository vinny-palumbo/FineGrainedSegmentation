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
    python3 fashion.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import cv2
import skimage.draw
from sklearn.model_selection import train_test_split

# Import Mask RCNN
MASKRCNN_DIR = os.path.abspath("Mask_RCNN")
sys.path.append(MASKRCNN_DIR)  
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(MASKRCNN_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(MASKRCNN_DIR, "logs")

# annotation files
LABELS_JSON_FILENAME = "label_descriptions.json"
ANNOTATIONS_CSV_FILENAME = "train.csv"

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


def load_datasets():
    
    # Get fashion labels from json file
    label_json_path = os.path.join(args.dataset, LABELS_JSON_FILENAME)
    with open(label_json_path) as f:
        label_descriptions = json.load(f)
    label_names = [x['name'] for x in label_descriptions['categories']]
    
    # read annotations CSV file into a pandas dataframe
    annotations_csv_path = os.path.join(args.dataset, ANNOTATIONS_CSV_FILENAME)
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
    dataset_train = FashionDataset(args.dataset, df_train, label_names)
    dataset_train.prepare()

    # load validation dataset
    dataset_val = FashionDataset(args.dataset, df_val, label_names)
    dataset_val.prepare()
    
    return dataset_train, dataset_val
    

def train(model):
    """Train the model."""
    
    # load train and valid datasets
    dataset_train, dataset_val = load_datasets()
    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


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
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect fashion items on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to detect fashion items on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image or args.video,\
               "Provide --image or --video to detect fashion items"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

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
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

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
        detect(model, image_path=args.image, video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
