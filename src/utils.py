import cv2
import colorsys
import os
import json
import random
import numpy as np

from Mask_RCNN.mrcnn import visualize


def get_labels(dataset, json_file):

    ''' Get fashion labels from json file '''
    
    label_json_path = os.path.join(dataset, json_file)
    with open(label_json_path) as f:
        label_descriptions = json.load(f)
    label_names = [x['name'] for x in label_descriptions['categories']]
    
    return label_names


def get_labels_colors(label_names):

    ''' Generate random (but visually distinct) colors for each class label '''
    
    hsv = [(i / len(label_names), 1, 1.0) for i in range(len(label_names))]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(42)
    random.shuffle(colors)

    return colors
    

def apply_masks_on_image(image, r, colors):
    
    ''' Applies masks of each detected object in an image '''
    
    # loop over of the detected objects in the image, and apply masks
    for i in range(0, len(r["rois"])):

        # extract the class ID and mask for the current detection, then
        # grab the color to visualize the mask (in BGR format)
        mask = r["masks"][:, :, i]
        classID = r["class_ids"][i]
        color = colors[classID][::-1]

        # visualize the pixel-wise mask of the object
        image = visualize.apply_mask(image, mask, color, alpha=0.5)
        
    return image


def draw_bbox_and_labels_on_image(image, r, colors, class_names):
    
    ''' Draws bounding boxes, class labels, and scores of each detected object in an image '''
    
    # convert the image back to BGR so we can use OpenCV's drawing functions
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # loop over of the detected objects in the image, and
    # draw detected objects bounding boxes and labels
    for i in range(0, len(r["rois"])):
        
        # extract the bounding box information, class ID, label, predicted
        # probability, and visualization color
        (startY, startX, endY, endX) = r["rois"][i]
        classID = r["class_ids"][i]
        color = [int(c) for c in np.array(colors[classID]) * 255]
        label = class_names[classID]
        score = r["scores"][i]
            
        # draw the bounding box, class label, and score of the object
        font_size = max(0.6, (.6 / 1200) * image.shape[0])
        thickness_size = int(max(2, (2 / 1200) * image.shape[0]))
        
        cv2.rectangle(image, (startX, startY), (endX, endY), color, thickness_size)
        
        text = "{}: {:.3f}".format(label, score)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_size, color, thickness_size)
        
    return image