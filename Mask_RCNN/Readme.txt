This repo is cloned from https://github.com/matterport/Mask_RCNN. 

Although dataset used to train on nuclei dataset which can be downloaded at : 
	https://drive.google.com/open?id=19HTlerF4LD_06GUc9-ckhVvunryY4duP

Trained weights can be downloaded from here:
	https://drive.google.com/open?id=1-0XIcLdHGJyDEozY3WvK5JIBBpDACy-e

	Model was trained for 25 epochs. So weights file would contain 4 GB worth of weights. Download accordingly



Folder structure:

	1. logs - folder containing trained weights. Currently trained to 20 epochs.Can be trained further from this checkpoint.
	2. mrcnn - containing supporting code including model initialization file
	3. nuclei_datasets - download training dataset and unzip here as it is
	4. results - resulting images with segmantation boundries drawn around nuclei. Test images part of above dataset(stage1_test)
	5. inspect_nucleus_model.ipynb - compute mean average precision(mAP) on 25 and 100 test and train samples respectively and plot the graph for each
	6. map_tracker_test.txt, map_tracker_train.txt - files containing raw mAP values
	7. nucleus.py - file containing training and testing code


Code Usage:


	Train a new model starting from ImageNet weights using train dataset (which is stage1_train minus validation set)

	1. python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet


	Train a new model starting from specific weights file using the full stage1_train dataset

	1. python3 nucleus.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5


	Resume training a model that you had trained earlier

	1. python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last


	Generate submission file from stage1_test images

	1. python3 nucleus.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=<last or /path/to/weights.h5>


Note: 
	1. All training weights are generated using Google Colab. Big thanks to Google for providing GPUs for free !!!!
	2. Thanks to maker of this repo for simplifying the process of training and testing. 