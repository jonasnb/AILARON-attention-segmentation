import cv2
import numpy as np
import os
from PIL import Image

#img= cv2.imread(r"C:\Users\jonasnb\Documents\pytorch_smp\out_cv\mog2_16.jpg")

mog2_img_names=os.listdir(r'C:\Users\jonasnb\Documents\pytorch_smp\background_opencv\paper_masks')

os.chdir(r"C:\Users\jonasnb\Documents\pytorch_smp\background_opencv\paper_dilation_out")

dirpath=r"C:\Users\jonasnb\Documents\pytorch_smp\background_opencv\paper_masks"

kernel_1=np.ones((10,10),np.uint8)
kernel_2=np.ones((20,20),np.uint8)
for mog2_name in mog2_img_names:
    
    filepath=os.path.join(dirpath, mog2_name)

    mog2_image=Image.open(filepath).convert("RGB")
    open_cv_image = np.array(mog2_image) 
    #opening=cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
    #closing= cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel_1)
    dilation=cv2.dilate(open_cv_image,kernel_1,iterations = 1)
    erotion=cv2.erode(dilation, kernel_2, iterations=1)

    out=Image.fromarray(erotion)
    out.save(mog2_name)

