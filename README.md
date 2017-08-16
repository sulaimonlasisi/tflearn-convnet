This program intends to simplify image classification using TFLearn Convoluted Neural Networks.
For more information about TFLearn and the parameters used here, see: http://tflearn.org/

Dependencies:
All these dependencies have to be installed before you can use the module.
Use the pip install command in front of each module to install.
ImageSlicer -- pip install image_slicer
TFLearn -- pip install tflearn
TensorFlow -- pip install tensorflow

How to Use:
  
  Training: 
  
  After cloning and installing dependencies, run below command with the following options:  
    images_folder is the folder containing all the data - it resides in the same directory as main.py    
    list_of_classes is a list of words e.g. (car bike) - each word represents a class that has a folder of images    
    each class is in images_folder    
    Optional flags that may be used in the training process:    
      --mirror: type=int, default = 1. Decides if images are flipped or not. Useful when samples are limited
      --split: type=int, default = 1. Decides if images are split or not.
      --crop': type=int, default = 32. If image is split, how many per page.
      --size: type=int, default = 256. Size used for image when passed into CNN
      --epoch: type=int, default = 100. Number of epochs to run
      --batches: type=int, default = 96. Number of images per iteration
      --id: default = 'cnn'. ID used to save model upon completion
      --test: type=float, default = 0.25. Fraction of samples to use for validation
      --accuracy: type=float, default = 0.9. Accuracy at which session is saved to best checkpoint path
      
    Example usage: python learn.py images_folder car bike --split 0 --flip 1 --test 0.1        
    This will run for a while and save the CNN to specified folder.
  
  Testing:

  After completing a training session, the checkpoint ID of the preferred session (usually the session with highest accuracy) to use is available in the list of checkpoints. The ID is usually the accuracy of the session, so when in doubt, choose the checkpoint with the highest accuracy.
  To run a test, 
    Save all images in the format "class_name (1).jpg" in a folder e.g. test_images_folder
    Save list of classes in a .txt file with each class on a separate line
    test_images_folder and .txt should be in the same directory as test.py
    Save classes in same order as you did when you entered them in command line to train
    Execute command: "python test.py checkpoint_id classes_file.txt test_images_folder"
