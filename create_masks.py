import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import matplotlib.image as mpimg
import glob 
from PIL import Image
from skimage.draw import polygon2mask, polygon, polygon_perimeter
import ast

os.chdir(r"C:\Users\jonasnb\Documents\test2\Pytorch-UNet-master\data\masks\train_masks_aug..")
data=pd.read_csv(r'C:\Users\jonasnb\Documents\test2\Pytorch-UNet-master\data\masks\csv_all\combined_csv_all.csv')
imgs=data['filename'].unique()

def gen_image(img):
    #shape = (2050,2448)  # image shape
    shape = (2448,2050)  # image shape

    imgp = np.full(shape, 0)  # fill a n*d matrix with '.'
    try:
        for index, row in img.iterrows():
            txt=row['region_shape_attributes']
            dictionary=ast.literal_eval(txt)
            points=(dictionary['all_points_x'],dictionary['all_points_y'])

            rr, cc = polygon(*points, shape=shape)
            imgp[rr, cc] = 255
    except:
        #imgp = np.full(shape, 0)
        for i in range(20):
            print('empty image')
    return imgp


df=data
for img_name in imgs:
    labels= df.loc[df['filename'] == img_name]
    i=gen_image(labels
    new_i = i.transpose()
    img_name=re.sub('.bmp$', '', img_name)
    im = Image.fromarray(new_i)
    im.save(img_name+"original_mask.gif")
    
    i2=np.flipud(new_i)
    im=Image.fromarray(i2)
    im.save(img_name+"vertical_mask.gif")
    
    i3=np.fliplr(new_i)
    im=Image.fromarray(i3)
    im.save(img_name+"horiz_mask.gif")
    
    i4=np.flipud(new_i)
    i5=np.fliplr(i4)
    im=Image.fromarray(i5)
    im.save(img_name+"both_mask.gif")
    
