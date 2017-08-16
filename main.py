import os, shutil, argparse
from glob import glob
from helpers import prepare_images, cnn
import collections


'''
Main file that does the following processes needed to train images.
Some steps can be skipped based on the amount of samples available
To skip a step, just comment it out.
'''

'''
Step 1 - Parse arguments to process images and classify with
'''
parser = argparse.ArgumentParser(description='Preprocess images and train classifiers.')
parser.add_argument('images_folder', help='Folder where images to be classified reside')
parser.add_argument('classes', nargs = '*', help='List of classes with separate folder of images residing in images_folder')
parser.add_argument('--mirror', type=int, default = 1, help='Choose 1 to flip images. Flip by default. Any other value does npt flip.')
parser.add_argument('--split', type=int, default = 1, help='Choose 1 to split images. Split by default. Any other value does npt split.')
parser.add_argument('--crop', type=int, default = 32, help='If splitting images, number of pages to split each image page into')
parser.add_argument('--size', type=int, default = 256, help='Size of image relevant to classification')
parser.add_argument('--epoch', type=int, default = 100, help='Number of epochs to run')
parser.add_argument('--batches', type=int, default = 96, help='Size of each batch through iteration')
parser.add_argument('--id', default = 'cnn', help='Label for the CNN that will be created')
parser.add_argument('--test', type=float, default = 0.25, help='Fraction of samples to use for validation')
parser.add_argument('--accuracy', type=float, default = 0.9, help='Accuracy at which session is saved to best checkpoint path')
args = parser.parse_args()

training_args = collections.OrderedDict()
training_args['size'] = args.size
training_args['epoch'] = args.epoch
training_args['batches'] = args.batches
training_args['id'] = args.id
training_args['test'] = args.test
training_args['accuracy'] = args.accuracy


'''
Step 2 - Flip Images Horizontally
Because of the asymmetry of print images, flipping horizontally
- creates a different image without losing the defect type.
Only do this if you don't have enough images to train your classifier.
Do this for each class of image present in the images_folder
-- There should be a step to clean out the mirror images later. Yet to be implemented
'''
image_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.images_folder)
if args.mirror == 1:
  for cls in range(len(args.classes)):    
    class_folder = os.path.join(image_dir, args.classes[cls])
    prepare_images.flip_images(cls, class_folder)


'''
Step 3 - Split each print image into a number of images that suit 
your purpose and sample size.
Only do this if you don't have enough images to train your classifier.
Default number of splits per image is 32.
Do this for each class of image present in the images_folder
-- There should be a step to clean out the split images later. Yet to be implemented
'''
if args.split == 1:
  for cls in range(len(args.classes)):
    #Include the number of splits as the fourth argument if 
    #you want to split it into any number other than 32   
    prepare_images.split_images(cls, image_dir, args.classes[cls], args.crop)



'''
Step 4 - Preprocess images to be used in classification
Change file size to suit classification need
Include file size as parameter, if not provided, use 256

If there are directories of split images, use those. Otherwise, use
all directories in the folder
'''
dirlist = [ item for item in os.listdir(image_dir) if os.path.isdir(image_dir) ]
if any(folder.split('_')[0] == 'split' for folder in dirlist):
  folders = [ item for item in dirlist if item.split('_')[0] == 'split' ]
else:
  folders = dirlist
training_args['folders'] = folders
training_args['image_dir'] = image_dir
cnn.train_cnn(training_args)
'''
Step 5 - Delete all the splits and flips
'''
if any(folder.split('_')[0] == 'split' for folder in folders):  
  for item in folders:
    shutil.rmtree(os.path.join(image_dir, item))
for cls in args.classes:
  for f in glob(os.path.join(image_dir, cls,'*mirr*')):
    os.remove(f)
