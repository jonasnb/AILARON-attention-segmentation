import os
from PIL import Image

dilation_out_dir=r'C:\Users\jonasnb\Documents\pytorch_smp\background_opencv\removed_bad_dilation_2'
dilation_out=os.listdir(dilation_out_dir)

train_dir = r"C:\Users\jonasnb\Documents\data_example\RAW_train"
train =os.listdir(train_dir)

mask_save_dir = r"C:\Users\jonasnb\Documents\pytorch_smp\background_opencv\new_data_set_2\masks"
img_save_dir = r"C:\Users\jonasnb\Documents\pytorch_smp\background_opencv\new_data_set_2\imgs"

for i in range(len(dilation_out)):
    for j in range(len(train)):
        id=dilation_out[i].removesuffix("_clas_mask.gif")
        if id in train[j]:
            mask_load_path=os.path.join(dilation_out_dir, dilation_out[i])
            mask=Image.open(mask_load_path)

            mask_save_path=os.path.join(mask_save_dir,dilation_out[i])
            mask.save(mask_save_path)

            img_load_path= os.path.join(train_dir, train[j])
            img=Image.open(img_load_path)

            img_name=train[j].removesuffix(".bmp")
            img_name=img_name+"_clas.bmp"
            img_save_path= os.path.join(img_save_dir, img_name)
            img.save(img_save_path)

