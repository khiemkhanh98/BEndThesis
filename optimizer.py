
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

l2=1e-4
dp=0.2


def get_link(layer, base_ch, growth_rate, grmul):
    if layer == 0:
        return base_ch, 0, []
    out_channels = growth_rate
    link = []
    
    for i in range(10):
        dv = 2 ** i
        if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
    
    out_channels = int(int(out_channels + 1) / 2) * 2
    
    return out_channels, link

def conv(x, depth, kernel_size=3, strides=1, padding = 'same',trainable=True):
    return tf.keras.layers.Conv2D(depth, (kernel_size,kernel_size), (strides,strides), padding= padding,
                                  kernel_initializer = tf.initializers.he_uniform(seed = 1),kernel_regularizer=tf.keras.regularizers.L2(l2),
                                  trainable=trainable)(x)

def batch_norm(x, trainable=True):
    return tf.keras.layers.BatchNormalization(fused=True,trainable=trainable)(x)

def relu(x):
    return tf.nn.relu(x)

def resize(x, size):
    return tf.image.resize(x, size)

def drop_out(x):
    return tf.keras.layers.Dropout(dp)(x)

def avg_pool(x, pool_size = 2, strides = 2):
    return tf.keras.layers.AveragePooling2D(pool_size, strides, padding = 'same')(x)

def conv_layer(x, depth, kernel_size=3, strides=1,trainable=True,dropout=True):
    x = conv(x, depth = depth, kernel_size=kernel_size, strides=strides,trainable=trainable)
    x = batch_norm(x)
    x = relu(x)
    if dropout:
        x = drop_out(x)   
    return x

def hardblock(x, in_channels, growth_rate, grmul, n_layers,trainable=True,dropout=True):
    layers = [x]
    out_layers = []
    out_channels = 0
    
    for i in range(n_layers):    
        links = []
        out_ch, link = get_link(i+1, in_channels, growth_rate, grmul)
    
        for i in link:
            links.append(layers[i])

        x = tf.concat(links, axis = -1)
        layers.append(conv_layer(x, out_ch,trainable=trainable,dropout=dropout))
    
    for i in range(len(layers)):
        if (i == len(layers)-1) or (i%2 == 1):
            out_layers.append(layers[i])  
    x = tf.concat(out_layers, axis = -1)
    return x

def transition_up(x, skip_connection):
    x = resize(x, (tf.shape(skip_connection)[1], tf.shape(skip_connection)[2]))
    x = tf.concat([x, skip_connection], axis = -1)
    return x

def se_block(x,trainable=True):
    num_reduced_filters = tf.cast(x.shape[-1]*0.25,tf.int32).numpy()
    se_tensor = layers.GlobalAveragePooling2D()(x)
    se_tensor = layers.Reshape((1, 1, x.shape[-1]))(se_tensor)
    se_tensor = layers.Conv2D(num_reduced_filters, 1, activation=tf.keras.activations.swish, padding='same',trainable=trainable,
                              kernel_initializer = tf.initializers.he_uniform(seed = 1),kernel_regularizer=tf.keras.regularizers.L2(l2))(se_tensor)
    se_tensor = layers.Conv2D(x.shape[-1],1,activation='sigmoid',padding='same',trainable=trainable,
                              kernel_initializer = tf.initializers.he_uniform(seed = 1),kernel_regularizer=tf.keras.regularizers.L2(l2))(se_tensor)
    x = layers.multiply([x, se_tensor])   
    return x

def hardnet(img):
    encoder_trainable_rgb=True
    encoder_trainable_depth = True
    depth = img[:,:,:,3:]
    img = img[:,:,:,:3]
    first_ch  = np.asarray([16,24,32,48])
    depth_conv1 = np.asarray([64, 96, 160, 224, 320])
    grmul = 1.7
    gr       = np.asarray([  10,16,18,24,32])
    n_layers = [4, 4, 8, 8, 8]
    up_sample = np.asarray([126,238,374,534])
    skip_connections = []
    skip_connections_d = []
    blks = len(n_layers)
    
    x = img
    x = conv_layer(x, first_ch[0], strides = 2,trainable=encoder_trainable_rgb)
    x = conv_layer(x, first_ch[1],trainable=encoder_trainable_rgb)
    x = conv_layer(x, first_ch[2], strides = 2,trainable=encoder_trainable_rgb)
    x = conv_layer(x, first_ch[3],trainable=encoder_trainable_rgb)

    
    x_d = depth
    x_d = conv_layer(x_d, first_ch[0], strides = 2,trainable=encoder_trainable_depth)
    x_d = conv_layer(x_d, first_ch[1],trainable=encoder_trainable_depth)
    x_d = conv_layer(x_d, first_ch[2], strides = 2,trainable=encoder_trainable_depth)
    x_d = conv_layer(x_d, first_ch[3],trainable=encoder_trainable_depth)
    
    ch = first_ch[3]
    ch_d = first_ch[3]
    for i in range(blks):
        x = hardblock(x, ch, gr[i], grmul, n_layers[i],trainable=encoder_trainable_rgb)
        x_d = hardblock(x_d, ch_d, gr[i], grmul, n_layers[i],trainable=encoder_trainable_depth)
        
        if i < blks-1:
            skip_connections.append(tf.math.add(se_block(x,trainable=encoder_trainable_rgb),se_block(x_d,trainable=encoder_trainable_depth))) 
            #skip_connections.append(x) 
        x = conv_layer(x, x.shape[-1], kernel_size = 1,trainable=encoder_trainable_rgb)
        x_d = conv_layer(x_d, x_d.shape[-1], kernel_size = 1,trainable=encoder_trainable_depth)
        if i < blks-1:
            x = avg_pool(x)
            x_d = avg_pool(x_d)
        ch = x.shape[-1]
        ch_d = x_d.shape[-1]
    
    x = tf.math.add(se_block(x,trainable=encoder_trainable_rgb),se_block(x_d,trainable=encoder_trainable_depth))

    n_blocks = blks-1
    for i in range(n_blocks-1,-1,-1):
        skip = skip_connections.pop()
        x = transition_up(x, skip)

        cur_channels_count = x.shape[-1]//2

        x = conv_layer(x, cur_channels_count, kernel_size=1,dropout=False)
        x = hardblock(x, cur_channels_count, gr[i], grmul, n_layers[i],dropout=False)

    x = conv(x, depth = 7, kernel_size = 1)
    
    return x

input = tf.keras.layers.Input(shape=(480,640,6))
output = hardnet(input)
model = tf.keras.Model(inputs=input, outputs=output)

model.load_weights('weights_frauas_7classes.h5')
model.save('model_frauas_7classes')

print('Converting to TF-TRT FP16...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16,
                                                               max_workspace_size_bytes=8000000000)

converter = trt.TrtGraphConverterV2(input_saved_model_dir='model_frauas_7classes',
                                    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='optimized_model_frauas_7classes')


'''
import time
saved_model_loaded = tf.saved_model.load('final_model_TFTRT_FP16', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
while 1:
	st = time.time()
	infer(tf.ones((1,480,640,6)))
	print(time.time()-st)
'''
