##scripts to convert scenenet to tfrecords

import tensorflow as tf
import glob
import os
import numpy as np

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def image_example(p_rgb,p_label,p_x,p_y,p_z):
    rgb_string = open(p_rgb, 'rb').read()
    label_string = open(p_label, 'rb').read()
    label = tf.image.decode_png(label_string,dtype = tf.uint8).numpy()
    x_string = open(p_x, 'rb').read()
    y_string = open(p_y, 'rb').read()
    z_string = open(p_z, 'rb').read()
    total_pixel = 240*320
    weight = []
    for i in range(14):
        pixel = np.sum(label==i)
        weight.append(float(pixel)/float(total_pixel)+1e-7)
    weight = (np.median(weight)/np.array(weight)).tolist()
    feature = {
        'rgb': _bytes_feature(rgb_string),
        'label': _bytes_feature(label_string),
        'x': _bytes_feature(x_string),
        'y': _bytes_feature(y_string),
        'z': _bytes_feature(z_string),
        'weight':_float_feature(weight)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

##since scenenet dataset is divided into 17 smaller parts
folder = '/home/tilab/segmentation/data/scenenet/'
for i in range(17):
    os.system(f'unzip {folder}'+f'scene{i}.zip') 
    rgb = sorted([f for f in glob.glob(folder + f'scene{i}/{i}/rgb/*')])
    label = sorted([f for f in glob.glob(folder + f'scene{i}/{i}/label/*')])
    x = sorted([f for f in glob.glob(folder + f'scene{i}/{i}/x/*')])
    y = sorted([f for f in glob.glob(folder + f'scene{i}/{i}/y/*')])
    z = sorted([f for f in glob.glob(folder + f'scene{i}/{i}/z/*')])
    cnt=0
    record_file = home + f'train_scenenet_{i}.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for p_rgb,p_label,p_x,p_y,p_z in zip(rgb,label,x,y,z):
            tf_example = image_example(p_rgb,p_label,p_x,p_y,p_z)
            writer.write(tf_example.SerializeToString())
        writer.close()

        
##create TFRecords for validation set
os.system(f'unzip {folder}' + 'valscene.zip') 
rgb = sorted([f for f in glob.glob(folder + 'valscene/data/rgb/*')])
label = sorted([f for f in glob.glob(folder + 'valscene/data/label/*')])
x = sorted([f for f in glob.glob(folder + 'valscene/data/x/*')])
y = sorted([f for f in glob.glob(folder + 'valscene/data/y/*')])
z = sorted([f for f in glob.glob(folder + 'valscene/data/z/*')])
cnt=0
record_file = home + 'tfrecords/val_scenenet.tfrecords'

with tf.io.TFRecordWriter(record_file) as writer:
    for p_rgb,p_label,p_x,p_y,p_z in zip(rgb,label,x,y,z):
        tf_example = image_example(p_rgb,p_label,p_x,p_y,p_z)
        writer.write(tf_example.SerializeToString())
    writer.close()
  

