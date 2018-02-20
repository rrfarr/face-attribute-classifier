#####################################################################################################
# testing VGG face model using a pre-trained model
# written by Zhifei Zhang, Aug., 2016
#####################################################################################################

from load_vgg_face_model import load_vgg_face_model
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

# build the graph
graph = tf.Graph()
with graph.as_default():
    # Set a Tensorflow placeholder
    input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])

    # Initialize the VGG network
    network, average_image = load_vgg_face_model('model/vgg-face.mat', input_maps)

# read sample image
img = imread('Aamir_Khan_March_2015.jpg', mode='RGB')
img = img[0:250, :, :]
img = imresize(img, [224, 224])
img = img - average_image



