import image_slicer
import os
from PIL.ImageOps import mirror 
from PIL import Image
import sys


def get_basename(filename):
    """Strip path and extension. Return basename."""
    return os.path.splitext(os.path.basename(filename))[0]

def split_images(images_folder, cls, num_splits):
  os.mkdir(os.path.join(images_folder, 'split_'+cls))
  for directory, subdirectories, files in os.walk(os.path.join(images_folder, cls)):
    for file in files:
      file_name = os.path.join(images_folder, cls, file)
      tiles = image_slicer.slice(file_name, int(num_splits), save = False)      
      image_slicer.save_tiles(tiles, directory=os.path.join(images_folder, 'split_'+cls),prefix=get_basename(file_name))
  #print('Class', idx, 'set of images split.')




'''
This module flips an image horizontally so it can create 
a slightly different duplicate to increase the sample size.
'''
def flip_images(cls, class_folder):
  #print(class_folder)
  flip_suffix = '_mirror.jpg'
  for directory, subdirectories, files in os.walk(class_folder):
    for file in files:
      file_name = os.path.join(class_folder, file)
      #print(file_name)
      im = Image.open(open(file_name, 'rb'))
      mirror_image = mirror(im)
      mirror_name = file.split('.')[0]+flip_suffix
      os.chdir(class_folder)
      mirror_image.save(mirror_name)
  #print('Class', cls, 'set of images flipped.')


def vertical_flip_image(cls, class_folder): 
  flip_suffix = '_top_bottom.jpg'
  for directory, subdirectories, files in os.walk(class_folder):
    for file in files:
      file_name = os.path.join(class_folder, file)
      im = Image.open(open(file_name, 'rb'))
      top_bottom_image = im.transpose(Image.FLIP_TOP_BOTTOM)
      top_bottom_name = file.split('.')[0]+flip_suffix
      os.chdir(class_folder)
      top_bottom_image.save(top_bottom_name)