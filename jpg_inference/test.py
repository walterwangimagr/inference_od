import numpy as np
from PIL import Image


# path = "/home/walter/nas_cv/walter_stuff/modular_dataset/from_scratch/training/images_default/denser_inside_skip10/1669233796150-modular-coral-v1-akl-0017:192.168.178.207.jpg"
# img_a = Image.open(path)
# crop = img_a.crop([20, 20, 300, 300])
# crop_array = np.array(crop)
# crop_img = Image.fromarray(crop_array)
# print(crop_array.shape)
# crop.show()



bg_img = Image.new('RGB', (960, 540))
bg_array = np.array(bg_img)
print(bg_array.shape)
bg_img.show()
# img = Image.fromarray(bg_image)
# image_array = np.array(img)
# print(image_array.shape)
# img.show()
