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

    # Freeze the graph
    graph.finalize()

    # read sample image
    img = imread('Aamir_Khan_March_2015.jpg', mode='RGB')
    img = img[0:250, :, :]
    img = imresize(img, [224, 224])
    img = img - average_image

    # Start the session
    with tf.Session() as session:
        try:
            # Extract the layer by name (named layers are keys in the network dict above)
            layer = graph.get_tensor_by_name("Relu_13:0") # We get predictions at this layer
            feed_dict = {input_maps: [img]}

            # run the image through the layer
            predictions = session.run(layer, feed_dict)
            predictions = np.squeeze(predictions)
            print(len(predictions))
        except:
            print("Layer not found")








