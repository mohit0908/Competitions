import os
import numpy as np
import random
import shutil

# Change training and validation directory
# Change split ratio as per requirement
# Training folder must contain structure:
	# |- category1 -> images
	# |- category2 -> images
	# |- category3 -> images
	# |- category4 -> images

# Same structure is replicated in validation folder

input_dir = 'dataset/augmented/training'
output_dir = 'dataset/augmented/validation'
train_ratio = 0.8


def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path)


def train_test_split(input_dir, output_dir, ratio):
	
	folder_dir = os.listdir(input_dir)
	for folder in folder_dir:
		makedirs(os.path.join(output_dir, folder))

		train_folder = os.path.join(os.getcwd(), input_dir, folder)
		images = os.listdir(train_folder)
		random_index = random.sample(range(0,len(images)), len(images))
		train_index, valid_index = random_index[0:int(len(random_index)*ratio)], random_index[int(len(random_index)*ratio):]
		

		for index in valid_index:
			shutil.move(os.path.join(input_dir, folder, images[index]), os.path.join(output_dir, folder, images[index]))
			print('Moved {}'.format(images[index]))


if __name__ == '__main__':
	train_test_split(input_dir, output_dir, train_ratio)
	print('Data splitted in training and validation sets')