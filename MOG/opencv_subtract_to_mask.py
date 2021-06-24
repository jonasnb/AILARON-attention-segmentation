

import numpy as np
import cv2
import os
import timeit
from PIL import Image




#img_names=os.listdir(r'C:\Users\jonasnb\Documents\data_example\RAW_train')
img_names=os.listdir(r'C:\Users\jonasnb\Documents\test2\Pytorch-UNet-master\data\RAW_may')

os.chdir(r"C:\Users\jonasnb\Documents\pytorch_smp\background_opencv\paper_masks")

cap = cv2.VideoCapture(r'C:\Users\jonasnb\Documents\pytorch_smp\background_opencv\project_may.avi')

#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
fgbg= cv2.createBackgroundSubtractorMOG2(detectShadows=False)

counter=0
start = timeit.default_timer()
for name in img_names:

    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    #cv2.imshow('frame',fgmask)
    #k = cv2.waitKey(30) & 0xff
    counter+=1
    name=name.removesuffix(".bmp")
    save_img_as = name+"_clas_mask.gif"

    im=Image.fromarray(fgmask)
    im.save(save_img_as)
    
    #if k == 27:
        #break

cap.release()
stop = timeit.default_timer()

print('Time: ', stop - start)  
cv2.destroyAllWindows()