##script to convert your custom dataset JSON label to PNG label

import labelme
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import numpy as np

folder = '/home/frauas/segmentation/data/custom/'
f = [os.path.split(file)[-1][:-5] for file in glob.glob(folder + 'json/*.json')]
for file in f:
    json_dir = folder + 'json/'
    label_file = labelme.LabelFile(filename=json_dir+file+'.json')
    img = labelme.utils.img_data_to_arr(label_file.imageData)
    
    #change to your own mapping, starting from 1, as Labelme has by default assigned regions without label as class 0 
    class_name_to_id = {'':1,'':2}
    
    lbl, _ = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=label_file.shapes,
                label_name_to_value=class_name_to_id,)
  
    dir = folder + 'label/'
    Image.fromarray(lbl.astype(np.uint8)).save(dir+file+'.png')