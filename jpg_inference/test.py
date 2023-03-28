import os 
import re 
import glob 
from PIL import Image 
import numpy as np

image_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/sonae_test/stack_bayer_white_balance/testset"
model_path = "/home/walter/nas_cv/walter_stuff/git/yolov5-master/yolo_n_modular/yolo5_nano_448/weights/no_nms_edgetpu.tflite"



image_files = glob.glob(f"{image_dir}/*.jpg")
for image_path in image_files:
    image = Image.open(image_path)
    image = image.resize((448,448))
    image.show()
    image_array = np.array(image, dtype=np.uint8)
    print(image_array.shape)  # Prints the dimensions of the array (height, width, channels)
    print(image_array.dtype) 
    break
