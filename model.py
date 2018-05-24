import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
from sklearn.model_selection import train_test_split #to split out training and testing data 
import tensorflow as tf
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
#popular optimization strategy that uses gradient descent 
#to save our model periodically as checkpoints for loading later
#what types of layers do we want our model to have?
#helper class to define input shape and generate training images given image paths & steering angles
from utils import INPUT_SHAPE, batch_generator, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, DATA_SHAPE, DATA_LENGTH, data_to_training_sample
#for command line arguments
import argparse
from PIL import Image

#for reading files
import os
import cv2

#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)

def batch_generator(x, y, batch_size):
    while (True):
        images = np.empty([batch_size, 288])
        outputs = np.empty([batch_size, 2])

        i = 0
        for index in np.random.permutation(x.shape[0]):

            images[i] = x[index]
            outputs[i] = y[index]
            i += 1
            if i == batch_size:
                break
        images = images.reshape(-1, 288)
        outputs = outputs.reshape(40, 2)

        yield images, outputs

def load_data(args):
    f = open(args.data_dir)

    all_data_x = np.empty(0)
    all_data_y = np.empty(0)

    raw_data = np.empty(288)

    i = 0
    for line in f:
        numbers = line.split(" ")

        if int(float(numbers[0])) != 288:
            speed = float(numbers[0])
            steering = float(numbers[1])
            x, y = data_to_training_sample(raw_data, speed, steering)
            print len(x)
            all_data_x = np.append(all_data_x, x)
            all_data_y = np.append(all_data_y, y)
            i = i+1
        else:
            raw_data = np.empty(288)
            for n in xrange(1, len(numbers)):
                raw_data[n - 1] = float(numbers[n])

    all_data_x = all_data_x.reshape((len(all_data_x) / 288, 288))
    all_data_y = all_data_y.reshape((-1, 2))

    print all_data_x
    print "Found, ", i, " examples"
    print all_data_x, all_data_y

    # 20% test set
    x_train, x_valid, y_train, y_valid = train_test_split(all_data_x, all_data_y, test_size=0.2, random_state=0)
    return x_train, x_valid, y_train, y_valid


def build_model(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Lambda(lambda x: x, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    #model.add(tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    # model.add(tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    # model.add(tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(tf.keras.layers.Dropout(args.keep_prob))
    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', input_shape=(288,)))
    model.add(tf.keras.layers.Dense(50))
    model.add(tf.keras.layers.Dense(20))
    model.add(tf.keras.layers.Dense(10))
    # model.add(tf.keras.layers.Dense(50, activation='relu'))
    # model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(2))
    model.summary()

    return model


def train_model(model, args, x_train, x_valid, y_train, y_valid):
    """np.array([np.random.random_sample(INPUT_SHAPE)]) the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate))

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    model.fit_generator(batch_generator(x_train, y_train, args.batch_size), #training data
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_queue_size=1,
                        validation_data=batch_generator(x_valid, y_valid, args.batch_size),#validation data
                        validation_steps=args.batch_size,
                        callbacks=[checkpoint],
                        verbose=1)

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    x_train, x_valid, y_train, y_valid = load_data(args)
    #build model
    model = build_model(args)
    #train model on data, it saves as model.h5 
    train_model(model, args, x_train, x_valid, y_train, y_valid)


if __name__ == '__main__':
    main()

