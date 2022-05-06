##script to convert 7-class frauas dataset JSON label to PNG label

import labelme
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import numpy as np

folder = '/home/frauas/segmentation/data/frauas_7classes/'

## we need 3class folder since we will copy the free space class from there
fra3_classes_folder = '/home/frauas/segmentation/data/frauas_3classes/' 

for i in range(5):
    f = [os.path.split(file)[-1][:-5] for file in glob.glob(folder + f'{i}/json/*.json')]
    for file in f:
        json_dir = folder + f'{i}/json/'
        fra3_classes_json_dir = fra3_classes_folder + f'{i}/json/'
        label_file = labelme.LabelFile(filename=json_dir+file+'.json')
        fra3_classes_label_file = labelme.LabelFile(filename=fra3_classes_json_dir+file+'.json')
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        class_name_to_id = {'human':1,'table':2,'chair':3,'robot':4,'backpack':5,'free':6,'patch':7}
        ## patch is in fact the other obstacles class, it is the class that we used to fill void holes
        lbl, _ = labelme.utils.shapes_to_label(
                    img_shape=img.shape,
                    shapes=label_file.shapes,
                    label_name_to_value=class_name_to_id,)
        fra3_classes_lbl, _ = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=fra3_classes_label_file.shapes,
                label_name_to_value=class_name_to_id,)
        lbl[fra3_classes_lbl==6] = 6 ##copy free space from 3classes dataset to the 7classes label
        lbl[lbl==7]=0 
        dir = folder + f'{i}/label/'
        Image.fromarray(lbl.astype(np.uint8)).save(dir+file+'.png')