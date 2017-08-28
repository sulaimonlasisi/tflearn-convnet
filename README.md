This program intends to simplify image classification using TFLearn Convoluted Neural Networks.
For more information about TFLearn and the parameters used here, see: http://tflearn.org/

Dependencies:
All these dependencies have to be installed before you can use the module.
Install modules as shown below:

ImageSlicer -- pip install image_slicer

TFLearn -- pip install tflearn

TensorFlow -- pip install tensorflow

How to Use:
  
  Training: 
  
  After cloning and installing dependencies, run below command with the following options:

    images_folder is the folder containing all the data - it resides in the same directory as main.py    
    
    list_of_classes is a list of words e.g. (car bike) - each word represents a class that has a folder of images    
      
    Optional flags that may be used in the training process:
        
      --horizontalflip: type=int, default = 1. Decides if images are horizontally flipped or not. Useful when samples are limited

      --verticalflip: type=int, default = 1. Decides if images are vertically flipped or not. Useful when samples are limited

      --split: type=int, default = 1. Decides if images are split or not.

      --splitlist: type - list. List of splits to make per image for each class. Must be same length as number of classes

      NOTE: The number of splits should be a value that still retains the desired feature after splitting

      --size: type=int, default = 256. Size used for image when passed into CNN (Depending on the number of samples, sizes larger than 512 can cause segmentation fault)

      --epoch: type=int, default = 100. Number of epochs to run

      --batches: type=int, default = 96. Number of images per iteration

      --id: default = 'cnn'. ID used to save model upon completion

      --test: type=float, default = 0.25. Fraction of samples to use for validation

      --accuracy: type=float, default = 0.9. Accuracy at which session is saved to best checkpoint path

      --mode: type: =int, default = 0. Color mode of images. 0 is for RGB, any other value is treated as grayscale
      
    Example usage: python main.py cars sedan vans suvs trucks  --split 1 --horizontalflip 1 --verticalflip 1 --test 0.1 --splitlist 16 8 16 16 --accuracy 0.9 --mode 1 --epoch 50  

    This will run for a while and save the CNN to same folder. Look for files beginning with cnn.tfl-xxxx. Use the highest value xxxx as the checkpoint_id argument for the test.py file when testing.
  
  Testing:

  After completing a training session, the checkpoint ID of the preferred session (usually the session with highest accuracy) to use is available in the list of checkpoints in the same folder. The ID is usually the accuracy of the session, so when in doubt, choose the checkpoint with the highest accuracy.
  To run a test, 
  
    Save all images in the format "class_name (1).jpg" in a folder e.g. test_images_folder

    Save list of classes in a .txt file with each class on a separate line

    test_images_folder and .txt should be in the same directory as test.py

    Save classes in same order as you did when you entered them in command line to train

    Execute command: python test.py checkpoint_id classes_file.txt test_images_folder --mode 1

  Sample folder structure:

  tflearn_convnet:.

  ├──────helpers

  ├──────images_folder

  │   ├──────bike

  │   └──────car
  
  └──────test_images_folder
