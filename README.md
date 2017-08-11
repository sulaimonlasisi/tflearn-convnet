This program intends to simplify image classification using TFLearn Convoluted Neural Networks.

Dependencies:
TFLearn
Scipy

How to Use:
	After pulling this from the repository, run the below command with the following options:
	images_folder is the folder containing all the data - it resides in the same directory as learn.py
	list_of_classes is a list of words e.g. (car bike) 
	  each word represents a class that has a folder containing images 
	  each class is in images_folder
	Optional flags that can be used in the training process:
	  --mirror: type=int, default = 1. Decides if images are flipped or not. Useful when samples are limited
	  --split: type=int, default = 1. Decides if images are split or not.
	  --crop': type=int, default = 32. If image is split, how many per page.
	  --size: type=int, default = 256. Size used for image when passed into CNN
	  --epoch: type=int, default = 100. Number of epochs to run
	  --batches: type=int, default = 96. Number of images per iteration
	  --id: default = 'cnn'. ID used to save model upon completion
	  --test: type=float, default = 0.2. Fraction of samples to use for validation
	  --accuracy: type=float, default = 0.9. Accuracy at which session is saved to best checkpoint path
	Example usage: python learn.py images_folder list_of_classes --split 0 --flip 1 --test 0.1

	This will run for a while and save the CNN to specified folder.