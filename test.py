from helpers import cnn
import argparse

parser = argparse.ArgumentParser(description='Test a given classifier using an existing checkpoint id')
parser.add_argument('ckpt', help='Checkpoint Id of test model')
parser.add_argument('classes_list_file', help='String representing path to file of classes. Each class is on a new line')
parser.add_argument('test_folder', help='String represending path to folder of test images. Images start with class name followed by space e.g. "classA 09.jpg"')
parser.add_argument('--mode', type=int, default = 0, help='Color mode of images. Default is RGB. Values other than 0 imply grayscale.')
args = parser.parse_args()
cnn.test_cnn(args.ckpt, args.classes_list_file, args.test_folder, args.mode)