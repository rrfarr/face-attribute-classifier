#####################################################################################################
# This script will be used to use the CELEB-A dataset and extract the features for all the images
# in this dataset. All the images will be passed through the VGG-Face CNN and we store the vector
# outputted at RELU 13:0
# written by Reuben Farrugia. February 2018
#####################################################################################################

from load_vgg_face_model import load_vgg_face_model
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

def parse_celeba_line(command):
    # Find the occurrance of .jpg
    idx = command.find('.jpg')+4

    # Parse the filename
    filename = command[0:idx]

    flag = int(command[idx+1:])

    return filename, flag

# Define the filename where the annotations are stored
anno_filename = 'DATASET/CELEB-A/Anno/list_attr_celeba.txt'

with open(anno_filename) as f:
    # Read line from the file
    annotations = f.readlines()

# Extract the annotation names
AttrNames = annotations[1].split()

# build the graph
graph = tf.Graph()
with graph.as_default():
    # Set a Tensorflow placeholder
    input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])

    # Initialize the VGG network
    network, average_image = load_vgg_face_model('model/vgg-face.mat', input_maps)

    # Freeze the graph
    graph.finalize()

    # Define the filename of the CELEB-A dataset
    celeba_filename = './DATASET/CELEB-A/Eval/list_eval_partition.txt'

    with open(celeba_filename) as f:
        # Read line from the file
        content = f.readlines()

    # Determine the number of images to be processed
    Nimgs = len(content)
    # Initialize lists
    feature_list = []
    filename_list = []
    traintest_list = []
    attribute_list = []
    for n in range(0,Nimgs):
        # Extract the content in each line
        line = content[n]
        # parse the line
        [filename, id] = parse_celeba_line(line)

        # Extract a line from the annotation file
        line = annotations[n+2]

        if line.find(filename) != -1:
            # Extract the attributes
            idx = line.find('.jpg')

            # Get a list of +1 -1 as attribute vector
            attr_vect = list(map(int, line[idx+4 :].split()))

            # Put the attributes in a list
            attribute_list.append(attr_vect)

        # Put the filename in the list
        filename_list.append(filename)

        # Put a flag indicating if it is a training, testing or validation image
        traintest_list.append(id)

        # Derive the full filename
        full_filename = 'DATASET/Celeb-A/Img/img_align_celeba/img_align_celeba/' + filename

        # read sample image
        img = imread(full_filename, mode='RGB')
        img = imresize(img, [224, 224])
        img = img - average_image

        # Start the session
        with tf.Session() as session:
            # Extract the layer by name (named layers are keys in the network dict above)
            layer = graph.get_tensor_by_name("Relu_13:0")  # We get predictions at this layer
            feed_dict = {input_maps: [img]}

            # run the image through the layer and extract the feature vector
            feature = session.run(layer, feed_dict)
            feature = np.squeeze(feature,axis=(0,1))

        # Append the feature to the feature list
        feature_list.append(feature)

        print(len(feature_list))

    # Close the file
    f.close()
    # Save the data which will be used for training
    np.save('DATA/filename',filename_list)
    np.save('DATA/features',feature_list)
    np.save('DATA/traintest',traintest_list)
    np.save('DATA/attribuges',attribute_list)