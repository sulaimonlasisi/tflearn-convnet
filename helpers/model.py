#tflearn CNN creation module
import os
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

def createCNN(args):

  size_entry_filter = 32
  size_mid_filter = 64
  filter_size = 3
  color_channels = 3
  model_main_activation = 'relu'
  model_exit_activation = 'softmax'
  max_pool_kernel_size = 2
  model_learning_rate = 0.001
  num_classes = args['num_classes']
  ###################################
  # Image transformations
  ###################################



  # normalisation of images
  img_prep = ImagePreprocessing()
  img_prep.add_featurewise_zero_center()
  img_prep.add_featurewise_stdnorm()

  # Create extra synthetic training data by flipping & rotating images
  img_aug = ImageAugmentation()
  img_aug.add_random_flip_leftright()
  img_aug.add_random_rotation(max_angle=25.)
  img_aug.add_random_blur(sigma_max=3.)


  ###################################
  # Define network architecture
  ###################################

  # Input is a image_size x image_size image with 3 color channels (red, green and blue)
  network = input_data(shape=[None, args['size'], args['size'], color_channels],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

  # 1: Convolution layer with 32 filters, each 3x3x3
  network = conv_2d(network, size_entry_filter, filter_size, activation=model_main_activation)

  # 2: Max pooling layer
  network = max_pool_2d(network, max_pool_kernel_size)

  # 3: Convolution layer with 64 filters
  network = conv_2d(network, size_mid_filter, filter_size, activation=model_main_activation)

  # 4: Convolution layer with 64 filters
  network = conv_2d(network, size_mid_filter, filter_size, activation=model_main_activation)

  # 5: Max pooling layer
  network = max_pool_2d(network, max_pool_kernel_size)
  
  

  # 6: Convolution layer with 64 filters
  network = conv_2d(network, size_mid_filter, filter_size, activation=model_main_activation)

  # 7: Convolution layer with 64 filters
  network = conv_2d(network, size_mid_filter, filter_size, activation=model_main_activation)

  # 8: Max pooling layer
  network = max_pool_2d(network, max_pool_kernel_size)
  
  
  # 9: Fully-connected 512 node layer
  network = fully_connected(network, 512, activation=model_main_activation)

  # 10: Dropout layer to combat overfitting
  network = dropout(network, 0.5)

  # 11: Fully-connected layer with two outputs
  network = fully_connected(network, num_classes, activation=model_exit_activation)

  # Configure how the network will be trained
  acc = Accuracy(name="Accuracy")
  network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=model_learning_rate, metric=acc)

  
  #main_dir = os.path.dirname(os.path.dirname(os.getcwd()))
  # Wrap the network in a model object
  model = tflearn.DNN(network, tensorboard_verbose=0, max_checkpoints = 2,
    best_checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), args['id']+'.tfl.ckpt'), 
    best_val_accuracy = args['accuracy'])
  return model