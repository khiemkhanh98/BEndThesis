import tensorflow as tf
import time
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
import imagezmq
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='client info')
parser.add_argument('-client_url', type=str ,help='url displayed on ngrok (e.g. tcp://0.tcp.ngrok.io:13469)')
args = parser.parse_args()
client_url = str(args.client_url)

saved_model_loaded = tf.saved_model.load('optimized_model_frauas_7classes', tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
infer = saved_model_loaded.signatures['serving_default']

def get_img():
    depth = None
    rgb = None
    while (depth is None) or (rgb is None):
        name, jpg_buffer = image_hub.recv_jpg()
        image_hub.send_reply(b'OK')
        print(name)
        if name == 'depth':
            depth = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
            depth = np.reshape(depth,(240,320,3))
            depth = np.array(cv2.resize(depth,(640,480)))
            depth = depth/255
        if name == 'rgb':
            rgb = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
            rgb = np.reshape(rgb,(480,640,3))
            rgb = rgb/255
    return rgb, depth

def post_processing(output):
    output = output[0]
    output = tf.nn.softmax(output).numpy()
    ## if the model is not confident about any class then we consider it as uncertain
    mask = (output<0.5).all(axis=-1)  
    output = np.argmax(output,axis=-1)
    output[mask] = 7 ##uncertain pixel are marked as class 7
    output = output.astype(np.uint8)
    _,output = cv2.imencode(".png", output)
    return output

image_hub = imagezmq.ImageHub()
sender = imagezmq.ImageSender(connect_to=client_url)  
print('ready now and listenning to client')
while True:  
    st = time.time()
    rgb,depth = get_img()
    img = np.concatenate((rgb,depth),axis=-1)
    img= tf.expand_dims(img,axis=0)
    img = tf.cast(img,tf.float32)
    output = infer(img)['conv2d_130']
    output = post_processing(output)
    sender.send_jpg('output', output)
    print('total time: ', time.time()-st)
