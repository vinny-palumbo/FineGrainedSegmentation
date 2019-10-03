"""
Mask R-CNN
Configurations, data loading, and training code for Fashion dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Modified by Vincent Palumbo

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 train.py --dataset=/path/to/fashion/dataset --weights=coco

    # Train a new model starting from ImageNet weights
    python3 train.py --dataset=/path/to/fashion/dataset --weights=imagenet
    
    # Resume training on the last model that you had trained on
    python3 train.py --dataset=/path/to/fashion/dataset --weights=last
    
    # Resume training on a previous model that you had trained on 
    python3 train.py --dataset=/path/to/fashion/dataset --weights=/path/to/weights.h5
    
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# get this file's directory and add src/ to sys.path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.join(FILE_DIR, "..")
sys.path.append(SRC_DIR)
from utils import get_labels, load_weights

# Import Mask RCNN
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --models
DEFAULT_MODELS_DIR = os.path.join(SRC_DIR, "../models")

# necessary annotation files in the data
DATA_DIR = os.path.abspath("data")
LABELS_JSON_FILENAME = "label_descriptions.json"
ANNOTATIONS_CSV_FILENAME = "train.csv"


############################################################
#  Configurations
############################################################

class FashionConfig(Config):

    '''Configuration for training on the fashion dataset.
    Derives from the base Config class and overrides some values.
    '''
    
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
    
        ''' Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        '''
        
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
    
    ''' Load training and validation datasets '''
    
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


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect fashion items.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/fashion/dataset/",
                        help='Directory of the Fashion dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco', 'imagenet', or 'last'")
    parser.add_argument('--models', required=False,
                        default=DEFAULT_MODELS_DIR,
                        metavar="/path/to/models/",
                        help='Logs and checkpoints directory (default=models/)')
    args = parser.parse_args()

    # Validate arguments
    assert args.dataset, "Argument --dataset is required for training"
    assert args.weights, "Argument --weights is required for training"
    
    print("Dataset: ", args.dataset)
    print("Weights: ", args.weights)
    print("Models: ", args.models)

    # Configurations
    config = FashionConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.models)
    
    # Load weights
    load_weights(model, args.weights)

    # Train model
    train(model)

