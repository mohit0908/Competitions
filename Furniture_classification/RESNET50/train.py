import os
import time
from tqdm import tqdm
import numpy as np
import argparse
import tensorflow as tf

# Training parameters

img_size = 224

def batch_gen(path, batch_size):


        data_path = path

        train_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(data_path,
                                                                                     target_size = (img_size,img_size),
                                                                                     batch_size = batch_size,
                                                                                     classes = ['defective', 'nondefective'])

        valid_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('cropped_data/validation',
                                                                             target_size = (img_size,img_size),
                                                                             batch_size = 10,
                                                                             classes = ['defective', 'nondefective'])

        return train_batches, valid_batches


def create_model():

        image_input = tf.keras.layers.Input(shape=(224,224,3))
        model = tf.keras.applications.ResNet50(include_top = True, weights ='imagenet')

        num_classes = 2

        last_layer = model.layers[-2].output # grabing 2nd last layer output
        out = tf.keras.layers.Dense(num_classes, activation = 'softmax', name = 'output')(last_layer)
        custom_resnet_model = tf.keras.models.Model(inputs = model.input, outputs = out)
        for layer in custom_resnet_model.layers[:-2]:
                layer.trainable  = False

        custom_resnet_model.compile(optimizer = 'adadelta',loss = 'categorical_crossentropy',metrics = ['accuracy'])
        print('Custom resnet model created')

        return custom_resnet_model


def model_train(path, size, batch_size, epochs):

        train_batches, valid_batches = batch_gen(path, batch_size)

        custom_resnet_model = create_model()
        t1 = time.time()
        custom_resnet_model.fit_generator(train_batches,validation_data = valid_batches, steps_per_epoch = size//batch_size, epochs = epochs)
        t2 = time.time()
        print('Model trained. Time taken per epoch:', (t2-t1)/epochs)
        custom_resnet_model.save('./ckpt/resnet_model.h5')
        print('Model saved')



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', help = 'path of training data.')
        parser.add_argument('--size', help = 'total number of training samples', type = int)
        parser.add_argument('--batch', help = 'number of samples per training batch', default = 64, type = int)
        parser.add_argument('--epochs', help = 'number of epochs for training', default = 5, type = int)


        args = vars(parser.parse_args())

        model_train(args['path'], args['size'], args['batch'], args['epochs'])


# Usage 
# python3 resnet.py --path <path of training data(data separated in class folders)> --size <dataset size> --batch <batch_size> --epochs <no_of_epochs> 
