#parsing command line arguments
import argparse
#decoding camera images
import os, cv2
#high level file operations
#matrix math
import numpy as np
#real-time server
import tensorflow as tf

#helper class
import utils
from utils import IMAGE_WIDTH, IMAGE_HEIGHT

#init our model and image array as empty
model = None
prev_image_array = None


#and a speed limit
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model

    model = tf.keras.models.load_model(args.model)
    speed = model.predict(np.array([np.empty([288])]), batch_size=1)
    print speed

