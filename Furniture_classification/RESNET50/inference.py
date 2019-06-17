import argparse
import time
import tensorflow as tf
import os
import cv2
import numpy as np
import create_model as cm

classes = ['chairs', 'curtains','sofas','wardrobes']


def calculate_accuracy(y, y_):
	matching_labels = np.count_nonzero(y_ == y)
	accuracy = matching_labels/len(y_)
	return accuracy

def inference(path, weights_file):

	category = path.split('/')[-1]

	model = cm.create_model(weights_file)
	
	t1 = time.time()
	counter = 0
	prediction_list = []
	for files in os.listdir(path):
		img = cv2.imread(os.path.join(path, files))
		img = cv2.resize(img, (224,224))
		img = np.expand_dims(img, 0)
		prediction = model.predict(img)
		# prediction_label = classes[np.argmax(prediction)]
		prediction_list.append(np.argmax(prediction))
		# print(files, prediction_label)
		counter += 1

	# Ground label
	y = np.full((1, counter), classes.index(category))
	accuracy = calculate_accuracy(y, np.array(prediction_list))
	print('Accuracy:', accuracy)

	t2 = time.time()
	print('Average prediction time:', (t2-t1)/counter)




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--inference_data', help = 'path to test images')
	parser.add_argument('--checkpoint', help = 'pretrained weights file', default = 'None')

	args = vars(parser.parse_args())


	inference(args['inference_data'], args['checkpoint'])


# Usage:

# python inference.py --inference_data <inference data path> --checkpoint <weights file if any>

# E.g.
# python3 inference.py --inference_data ../dataset/augmented/validation/curtains --checkpoint ckpt/weights_09_1.17.hdf5
