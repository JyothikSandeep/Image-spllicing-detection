from tensorflow.keras.models import load_model
import numpy as np
model = load_model("C:/Users/Sandeep/Desktop/main project/model/image_splicing.h5")
# print(model)
from PIL import Image, ImageChops, ImageEnhance
import os
import itertools
import cv2

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file.jpg'
    ela_filename = 'temp_ela_file.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    # ela_image=np.uint8(ela_image)
    # cv2.imshow('img',ela_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # image.save('ela',ela_image)
    return ela_image
# real_image_path = "C:/Users/Sandeep/Desktop/projects/image-splicing-detection/static/uploads/Tp_D_CRN_M_N_nat10130_cha00086_11523.jpg"
# Image.open(real_image_path)
image_size = (128, 128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 85).resize(image_size)).flatten() / 255.0
# convert_to_ela_image(real_image_path, 85)
import numpy as np
path = "C:/Users/Sandeep/Desktop/projects/image-splicing-detection/static/uploads/Tp_D_CRN_M_N_nat10130_cha00086_11523.jpg"
x2 = prepare_image(path)
x2 = x2.reshape(-1, 128, 128, 3)
arr = model.predict(x2)
print(arr)
if(arr[0][0]>arr[0][1]):
    print("IMAGE IS TAMPERED")
else:
    print("IMAGE IS AUTHENTICATED")