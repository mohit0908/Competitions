import os
import time
from tqdm import tqdm
import numpy as np
import argparse
import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import create_model as cm

# Training parameters

img_size = 224
classes = ['chairs', 'curtains','sofas','wardrobes']

def batch_gen(train_path, valid_path,batch_size):

    train_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(train_path,
                                                                                 target_size = (img_size,img_size),
                                                                                 batch_size = batch_size,
                                                                                 classes = classes)

    valid_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(valid_path,
                                                                         target_size = (img_size,img_size),
                                                                         batch_size = batch_size,
                                                                         classes = classes)
    return train_batches, valid_batches


def model_train(train_path, valid_path, size, batch_size, epochs, weights_file):

        if not os.path.exists('ckpt'):
            os.makedirs('ckpt')

        train_batches, valid_batches = batch_gen(train_path, valid_path, batch_size)

        custom_resnet_model = cm.create_model(weights_file)
        
        # Define callbacks
        filepath = 'ckpt/weights_{epoch:02d}_{val_loss:.2f}.hdf5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir = './tensorboard_logs', histogram_freq=0, write_graph = True, write_images = False)
        csvlogger = tf.keras.callbacks.CSVLogger('log.csv', append = True, separator = ';')
        callback_list = [checkpoint, tensorboard, csvlogger]
        print('Callbacks Initiated')

        t1 = time.time()
        custom_resnet_model.fit_generator(train_batches,validation_data = valid_batches, steps_per_epoch = size//batch_size, epochs = epochs, callbacks=callback_list)
        t2 = time.time()
        print('Model trained. Time taken per epoch:', (t2-t1)/epochs)



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_path', help = 'path to training data.')
        parser.add_argument('--valid_path', help = 'path to validation data.')
        parser.add_argument('--size', help = 'total number of training samples', type = int)
        parser.add_argument('--batch', help = 'number of samples per training batch', default = 64, type = int)
        parser.add_argument('--epochs', help = 'number of epochs for training', default = 5, type = int)
        parser.add_argument('--checkpoint', help = 'pretrained weights file', default = 'None')

        args = vars(parser.parse_args())

        model_train(args['train_path'],args['valid_path'], args['size'], args['batch'], args['epochs'], args['checkpoint'])


# Usage 
# python3 train.py --train_path <path of training data(data separated in class folders)> --valid_path <path of validation data(data separated in class folders)> 
# --size <dataset size> --batch <batch_size> --epochs <no_of_epochs> --checkpoint <weights file if any>

# E.g.
# python3 train.py --train_path ../dataset/augmented/training --valid_path ../dataset/augmented/validation --size 7600 --batch 64 --epochs 10 --checkpoint ckpt/weights_03_0.19.hdf5
