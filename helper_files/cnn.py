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
from tflearn.data_utils import to_categorical
from tflearn.metrics import Accuracy
import scipy
import argparse
from .model import createCNN
import collections
import pdb



###################################
### Create Test/Train Split of Imported Images
###################################
def import_sample_objects(args):
  classes_file_paths = collections.OrderedDict()
  class_files = collections.OrderedDict()
  num_files = 0
  print(args['folders'])
  for folder in args['folders']:
    print(folder)
    classes_file_paths[folder] = os.path.join(args['image_dir'], folder, '*.*g') #this will capture png and jpg files
    class_files[folder] = sorted(glob(classes_file_paths[folder]))
    num_files+=len(class_files[folder])
  allX = np.zeros((num_files, args['size'], args['size'], 3), dtype='float64')
  ally = np.zeros(num_files)
  count = 0
  class_ctr = 0
  for key, value in class_files.items():
    print(key)
    for f in value:
      try:
        img = io.imread(f)
        new_img = imresize(img, (args['size'], args['size'], 3))
        allX[count] = np.array(new_img)
        ally[count] = class_ctr
        count += 1
      except:
        continue
    class_ctr+=1
  return allX, ally
  

###################################
# Prepare train & test samples
###################################
def split_and_categorize_samples(args, allX, ally):
  # test-train split   
  X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=args['test'])

  # encode the Ys
  Y = to_categorical(Y, len(args['folders']))
  Y_test = to_categorical(Y_test, len(args['folders']))
  return X, X_test, Y, Y_test

  

###################################
# Create, train and save model
###################################
def train_cnn(args):
  allX, ally = import_sample_objects(args)
  X, X_test, Y, Y_test = split_and_categorize_samples(args, allX, ally)

  #create cnn_model
  cnn_args = collections.OrderedDict()
  cnn_args['size'] = args['size']
  cnn_args['id'] = args['id']
  cnn_args['accuracy'] = args['accuracy']
  model = createCNN(cnn_args)

  ###################################
  # Train model for args['epoch'] epochs
  ###################################
  # Train it!
  model.fit(X, Y, n_epoch=args['epoch'], 
       shuffle=True, validation_set=(X_test, Y_test),
        show_metric=True, batch_size=args['batches'],
        snapshot_epoch=True,
        run_id=args['id'])
  print("Done with training")
  # Save model when training is complete to a file
  main_dir = os.path.dirname(os.path.dirname(os.getcwd()))
  model.save(os.path.join(main_dir, args['id']+'.tfl'))
  print('Network trained and saved as', os.path.join(main_dir, args['id']+'.tfl'))



def create_classes_list_from_file(file):
  with open(file) as f:
    classes_list = f.readlines()
    classes_list = [x.strip('\n') for x in classes_list]
  return classes_list

def create_prediction_metrics(model, size, image_dir, classes_list, metrics_array):
  test_count = 0
  #Crawl through directory and test all images in directory
  #Print the predicted class compared to the original class
  for directory, subdirectories, files in os.walk(image_dir):
    for file in files:
      img = scipy.ndimage.imread(os.path.join(directory, file), mode="RGB")
      img = scipy.misc.imresize(img, (size, size), interp="bicubic").astype(np.float32, casting='unsafe')
      prediction_label = model.predict_label([img])
      print("File  name: ", file, "label: ", classes_list[prediction_label[0][0]])
      metrics_array[classes_list.index(file.split(" ")[0])][2]+=1
      if classes_list[prediction_label[0][0]] == file.split(" ")[0]:
        metrics_array[prediction_label[0][0]][0]+=1
      metrics_array[prediction_label[0][0]][1] +=1
      test_count+=1

  #Calculate precition metrics for each class
  total_true_positives = 0
  total_all_positives = 0
  for idx, metric in enumerate(metrics_array):
    total_true_positives+=metric[0]
    total_all_positives+=metric[1]
    print(classes_list[idx], ": true positives -", metric[0], "all positives -", metric[1], "total in dataset is", metric[2])
    if metric[1] == 0:
      metric[1]+=1
    if metric[2] == 0:
      metric[2]+=1
    print("Precision :", np.divide(metric[0],metric[1]), "Recall :", np.divide(metric[0],metric[2]))



def test_cnn(ckpt_id, classes_list_file, test_folder):
  '''
  These variables are the same as those used in training and must be maintained
  to ensure consistent results. Before changing these variables, change them in 
  train_cnn before training.
  '''
  metrics_array = np.zeros([2, 3], dtype=int)
  cnn_args = collections.OrderedDict()
  cnn_args['size'] = 256
  cnn_args['id'] = 'cnn'
  cnn_args['accuracy'] = 0.9
  model = createCNN(cnn_args)
  model.load(cnn_args['id']+'.tfl.ckpt'+ckpt_id)
  classes_list = create_classes_list_from_file(classes_list_file)
  create_prediction_metrics(model, cnn_args['size'], test_folder, classes_list, metrics_array)

