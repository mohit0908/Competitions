# File description

1. scrape_weblinks.py:

	Run this file to download 240 images each from 4 categories - ['chairs', 'curtains','sofas','wardrobes']
	Chromedriver needs to be installed for running this script
	Data cleaning is required since some images doesn't belong to the category seached for
	Paths are mentioned in the file itself


2. augmentation_script.py

	Run this file to augment dataset 9X times as training data is less in number
	This script will create folder inside dataset folder and write augmented images as per classes
	Paths are mentioned in the file itself
	More augmentations can be added as per requirement


3. train_test_split.py

	Run this script to distribute data in training and validation sets.
	Train and validation directory are mentioned in file
	Split ratio is defined in file itself

4. Comparitive_Analysis.xlsx

	Table containing some comparitive aspects for ResNet50 and MobileNet

5. Comparative_Analysis.ipynb

	This iPython notebook contains plot based comparison of accuracy and loss metrics between ResNet50 and MobileNet

6. Dataset

	This folder contains training and validation splits of data. Dataset/augmented contains training and validation splits

7. RESNET50 and Mobilenet

	1. ckpt: folder containing pretrained weights
	2. tensorboard_logs: contains tensorboard events. This file is created while doing trainig
	3. create_model.py
		This file contains model creation code for individual models using base weights (imagenet) or pretrained ckpt weights if present
	4. train.py
		This file contains training code. Usage script is mentioned in the file itself.
	5. inference.py
		This file contains inference code for predicting class of images. Usage script is mentioned in the file itself.
	6. ckpt
		This contains weights file for inference
	7. log.csv
		This file is created while doing trainig and contains details about training, validation --> loss and accuracy
