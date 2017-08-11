"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""

from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy


###################################
### Import picture files 
###################################

def train_image(args):
  classes_file_paths = {}
  class_files = {}
  num_files = 0
  for folder in args['folders']:
    classes_file_paths[folder] = os.path.join(args['image_dir'], folder, '*g') #this will capture png and jpg files
    class_files[folder] = sorted(glob(classes_file_paths[folder]))
    num_files+=len(class_files[folder])
  print(num_files)
  allX = np.zeros((num_files, args['size'], args['size'], 3), dtype='float64')
  ally = np.zeros(num_files)
  count = 0
  for key, value in class_files.items():
    for f in value:
      try:
        img = io.imread(f)
        new_img = imresize(img, (args['size'], args['size'], 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
      except:
        continue
  ###################################
  # Prepare train & test samples
  ###################################

  # test-train split   
  X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=args['test'], random_state=int(args['test']*count))

  # encode the Ys
  Y = to_categorical(Y, len(args['folders']))
  Y_test = to_categorical(Y_test, len(args['folders']))


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

  ###################################
  # Define network architecture
  ###################################

  # Input is a image_size x image_size image with 3 color channels (red, green and blue)
  network = input_data(shape=[None, args['size'], args['size'], 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

  # 1: Convolution layer with 32 filters, each 3x3x3
  conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

  # 2: Max pooling layer
  network = max_pool_2d(conv_1, 2)

  # 3: Convolution layer with 64 filters
  conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

  # 4: Convolution layer with 64 filters
  conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

  # 5: Max pooling layer
  network = max_pool_2d(conv_3, 2)

  # 6: Convolution layer with 64 filters
  conv_4 = conv_2d(network, 64, 3, activation='relu', name='conv_4')

  # 7: Convolution layer with 64 filters
  conv_5 = conv_2d(conv_4, 64, 3, activation='relu', name='conv_5')

  # 8: Max pooling layer
  network = max_pool_2d(conv_5, 2)

  # 9: Fully-connected 512 node layer
  network = fully_connected(network, 512, activation='relu')

  # 10: Dropout layer to combat overfitting
  network = dropout(network, 0.5)

  # 11: Fully-connected layer with two outputs
  network = fully_connected(network, 2, activation='softmax')

  # Configure how the network will be trained
  acc = Accuracy(name="Accuracy")
  network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001, metric=acc)

  
  main_dir = os.path.dirname(os.path.dirname(os.getcwd()))
  # Wrap the network in a model object
  model = tflearn.DNN(network, tensorboard_verbose=0, max_checkpoints = 2, best_checkpoint_path = os.path.join(main_dir, args['id']+'.tfl.ckpt'), best_val_accuracy = args['accuracy'])

  ###################################
  # Train model for args['epoch'] epochs
  ###################################
  # Train it!
  model.fit(X, Y, n_epoch=args['epoch'], shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=args['batches'],
          snapshot_epoch=True,
          run_id=args['id'])
  print("Done with training")

  # Save model when training is complete to a file
  model.save(os.path.join(main_dir, args['id']+'.tfl'))
  print('Network trained and saved as', os.path.join(main_dir, args['id']+'.tfl'))

