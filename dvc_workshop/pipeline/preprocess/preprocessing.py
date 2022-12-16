from keras_preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img
from skimage import io 
import numpy as np
from PIL import Image
import os,glob

def main() :
    datagen = ImageDataGenerator(        
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

    image_directory = r'data/raw/Images'
    target_directory =r'data/processed/Images'
    SIZE = 256
    dataset = []
    my_images  = glob.glob(image_directory + "/**/*.jpg",recursive=True)

    target_exist = os.path.exists(target_directory)
    if not target_exist:
        # Create a new directory because it does not exist
        os.makedirs(target_directory)

    for img in my_images:
        if "DS_Store" in img: continue
        src_fname, ext = os.path.splitext(img) 
    

        img = load_img(img)

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        img_name = src_fname.split('/')[-1]

        i = 0
        for batch in datagen.flow (x, batch_size=1, save_to_dir = target_directory, 
                                save_prefix = img_name, save_format='jpg'):
            i+=1
            if i>3:
                break

if __name__ == "__main__":
    main()
