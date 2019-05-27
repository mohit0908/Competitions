import argparse
import time
import tensorflow as tf
import os
import cv2
import numpy as np


def inference(path):

	model = tf.keras.models.load_model('./ckpt/resnet_model.h5')
	
	print('Model loaded from checkpoint')

	t1 = time.time()
	counter = 0
	for files in os.listdir(path):
		img = cv2.imread(os.path.join(path, files))
		img = cv2.resize(img, (224,224))
		img = np.expand_dims(img, 0)
		prediction = model.predict(img)
		print(files, prediction)
		counter += 1

	t2 = time.time()
	print('Average prediction time:', (t2-t1)/counter)




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--data', help = 'path to test images')

	args = vars(parser.parse_args())


	inference(args['data'])