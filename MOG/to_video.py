# from: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
import cv2
import numpy as np
import os

 
#dirpath= r"C:\Users\jonasnb\Documents\pytorch_smp\to_video_imgs"
dirpath= r"C:\Users\jonasnb\Documents\test2\Pytorch-UNet-master\data\RAW_jan"
img_array = []

nr_to_include = 16   #decides how many images to include
counter= 0 
for filename in os.listdir(r'C:\Users\jonasnb\Documents\test2\Pytorch-UNet-master\data\RAW_jan'):
    
    filepath=os.path.join(dirpath, filename)


    
    img = cv2.imread(filepath)

    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    counter+=1
    #if (counter >= nr_to_include):
        #break
 
out = cv2.VideoWriter('project_jan.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()