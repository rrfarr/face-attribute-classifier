from load_vgg_face_model import load_vgg_face_model
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import os

max_batch_size = 3

print('initializing')

#Create files to enable restoring progress if program was abruptly terminated last time
if not os.path.isfile('DATA/temp.txt'):
    with open('DATA/temp.txt', 'w', encoding='utf-8') as f:
        pass

#Batch of images to pass through CNN together
image_batch = np.empty([ max_batch_size, 224, 224, 3 ])
image_batch_next_pos = 0

#Batch of data to save in temp.txt
info_batch = list()

#Get number of images that were already processed in a previous session
print('counting number of images already processed')
num_ready = 0
with open('DATA/temp.txt', 'r', encoding='utf-8') as f:
    for line in f:
        num_ready += 1
print(' ', num_ready)

#Create tensorflow graph
g = tf.Graph()
with g.as_default():
    print('loading VGG face model')
    img_in = tf.placeholder(tf.float32, [ None, 224, 224, 3 ])
    (model, average_image) = load_vgg_face_model('model/vgg-face.mat', img_in)
    features_layer = g.get_tensor_by_name("Relu_13:0") #CNN layer to extract features from
    g.finalize()
    
    print('starting image processing')
    with tf.Session() as sess:
        with open('DATASET/CELEB-A/Anno/list_attr_celeba.txt',    'r', encoding='utf-8') as attr_f, \
             open('DATASET/CELEB-A/Eval/list_eval_partition.txt', 'r', encoding='utf-8') as split_f:
            attr_lines = iter(attr_f)
            split_lines = iter(split_f)
            
            next(attr_lines) #Skip first line (contains number of rows)
            attr_names = next(attr_lines).split() #Get header row
            
            #Skip any previously processed rows
            print('skipping images already processed')
            for _ in zip(range(num_ready), attr_lines, split_lines): #zip will be as long as the shortest iterator so num_ready will determine when to stop
                pass
            
            for (attr_line, split_line) in zip(attr_lines, split_lines):
                attr_fields = attr_line.strip().split()
                split_fields = split_line.strip().split()
                assert (attr_fields[0] == split_fields[0]), 'Misaligned lines in attributes ({}) and partition files ({}).'.format(attr_fields[0], split_fields[0])
                
                #Add image info to batch for saving later after image features have been extracted (leave them as strings as they will be saved in a text file)
                filename = attr_fields[0]
                attributes = attr_fields[1:]
                split = split_fields[1]
                info_batch.append([ filename, split, attributes ])
                
                #Encode image and add it to batch
                full_filename = 'DATASET/Celeb-A/Img/img_align_celeba/img_align_celeba/'+filename
                encoded_img = imread(full_filename, mode='RGB')
                encoded_img = imresize(encoded_img, [ 224, 224 ])
                encoded_img = encoded_img - average_image
                image_batch[image_batch_next_pos] = encoded_img
                image_batch_next_pos += 1
                num_ready += 1
                
                #Save batch
                if image_batch_next_pos == max_batch_size:
                    features_batch = sess.run(features_layer, { img_in: image_batch })
                    with open('DATA/temp.txt', 'a', encoding='utf-8') as f:
                        for ((filename, split, attributes), features) in zip(info_batch, features_batch):
                            print(filename, split, ' '.join(attributes), ' '.join(str(f) for f in features), sep='\t', file=f)
                    image_batch_next_pos = 0
                    info_batch.clear()
                    print(num_ready, 'images ready')
            
            #Save any remaining data
            if image_batch_next_pos > 0:
                features_batch = sess.run(features_layer, { img_in: image_batch[:image_batch_next_pos] })
                with open('DATA/temp.txt', 'a', encoding='utf-8') as f:
                    for ((filename, split, attributes), features) in zip(info_batch[:image_batch_next_pos], features_batch[:image_batch_next_pos]):
                        print(filename, split, ' '.join(attributes), ' '.join(str(f) for f in features), sep='\t', file=f)
            print('all images processed')
del g
del model
del info_batch
del image_batch

#Transfer temp.txt to separate numpy files
print('saving final files')
with open('DATA/temp.txt', 'r', encoding='utf-8') as f:
    print('  filenames.npy')
    filenames = np.array([ line.strip().split('\t')[0] for line in f ], dtype=np.str_)
    np.save('DATA/filename.npy', filenames)
    del filenames
with open('DATA/temp.txt', 'r', encoding='utf-8') as f:
    print('  traintest.npy')
    splits = np.array([ line.strip().split('\t')[1] for line in f ], dtype=np.int8)
    np.save('DATA/traintest.npy', splits)
    del splits
with open('DATA/temp.txt', 'r', encoding='utf-8') as f:
    print('  attributes.npy')
    attributes = np.array([ line.strip().split('\t')[2].split(' ') for line in f ], dtype=np.int8)
    np.save('DATA/attributes.npy', attributes)
    del attributes
with open('DATA/temp.txt', 'r', encoding='utf-8') as f:
    print('  features.npy')
    features = np.array([ line.strip().split('\t')[3].split(' ') for line in f ], dtype=np.float32)
    np.save('DATA/features.npy', features)
    del features

print('ready!')
