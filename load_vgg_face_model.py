import tensorflow as tf
import numpy as np
from scipy.io import loadmat


def load_vgg_face_model(param_path, input_maps):
    # Load the matconvnet VGG face model
    data = loadmat(param_path)

    # read meta info
    meta = data['meta']
    classes = meta['classes']
    class_names = classes[0][0]['description'][0][0]
    normalization = meta['normalization']
    average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
    image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
    input_maps = tf.image.resize_images(input_maps, [image_size[0], image_size[1]])

    # read layer info
    layers = data['layers']
    current = input_maps
    network = {}
    # Initialize the layer counter
    lyr_idx = 1
    for layer in layers[0]:
        name = layer[0]['name'][0][0]
        layer_type = layer[0]['type'][0][0]
        if layer_type == 'conv':
            if name[:2] == 'fc':
                padding = 'VALID'
            else:
                padding = 'SAME'
            stride = layer[0]['stride'][0][0]
            kernel, bias = layer[0]['weights'][0][0]
            # kernel = np.transpose(kernel, (1, 0, 2, 3))
            bias = np.squeeze(bias).reshape(-1)
            conv = tf.nn.conv2d(current, tf.constant(kernel),
                                strides=(1, stride[0], stride[0], 1), padding=padding)
            current = tf.nn.bias_add(conv, bias)
            print(name, 'stride:', stride, 'kernel size:', np.shape(kernel))
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
            print(name)
        elif layer_type == 'pool':
            stride = layer[0]['stride'][0][0]
            pool = layer[0]['pool'][0][0]
            current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                     strides=(1, stride[0], stride[0], 1), padding='SAME')
            print(name, 'stride:', stride)
        elif layer_type == 'softmax':
            current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))
            print(name)
        # Put layers in the CNN model
        network[name] = current

        # Only consider the first 33 layers as done in the MATLAB script
        if (lyr_idx == 33):
            break

        # Increment the layer counter
        lyr_idx += 1

    return network, average_image