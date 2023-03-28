import numpy as np
from detectors import SSD_OD
from PIL import Image, ImageDraw
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
from bayer_to_RGB import bayer_to_rgb
import glob
import os
import re


def parse_objs(objs):
    """parse objs from 
    [Object(id=0, score=0.62890625, bbox=BBox(xmin=800, ymin=74, xmax=869, ymax=222))]
    to detections format 
    [xmin, ymin, xmax, ymax, score]

    Args:
        objs (list): list of objs

    Returns:
        np.array: shape(n, 5), n is the num of detecions, 5 is [xmin, ymin, xmax, ymax, score], if none, return an np.empty()
    """
    if len(objs) > 0:
        detections = []
        for obj in objs:
            xmin = obj.bbox.xmin
            ymin = obj.bbox.ymin
            xmax = obj.bbox.xmax
            ymax = obj.bbox.ymax
            score = obj.score
            detection = [xmin, ymin, xmax, ymax, score]
            detections.append(detection)

        return np.array(detections)
    else:
        return np.empty((0, 5))


bayer_dir = "/home/walter/big_daddy/nigel/mod_od_testsets/OD_test_small_product"

model_path = "/home/walter/nas_cv/object_detection/tensorflow/models/tpu_models/mobilenetv2_full_basket/walter_exp/default_bayer/exp47/export/exp47.tflite"

dataset_name = os.path.basename(bayer_dir)
model_name = os.path.basename(model_path)
model_name = re.sub(".tflite", "", model_name)

save_path = f"/home/walter/nas_cv/walter_stuff/git/modular-end2end-testing/od_eval/result/crop_map/{dataset_name}_{model_name}"
os.makedirs(save_path, exist_ok=True)


bayer_files = glob.glob(f"{bayer_dir}/*0016*.bayer_8")

objects_count = 0
total_images_count = len(bayer_files)

bg_img = Image.new('RGB', ((960, 540)))
bg_array = np.array(bg_img)


for bayer_file in bayer_files:
    bayer_data = np.fromfile(bayer_file, np.uint8).reshape((1080, 1920, 1))
    img = bayer_to_rgb(bayer_file)
    # img.show()

    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    common.set_input(interpreter, bayer_data)
    interpreter.invoke()
    objs = detect.get_objects(
        interpreter,
        score_threshold=0.5,
        image_scale=(2, 2))

    detections = parse_objs(objs)
    for detection in detections:
        xmin = max(0, int(detection[0]))
        ymin = max(0, int(detection[1]))
        xmax = min(960, int(detection[2]))
        ymax = min(540, int(detection[3]))
        bbox = np.array([xmin, ymin, xmax, ymax])

        img_crop = img.crop(bbox)
        crop_array = np.array(img_crop)
        bg_array[ymin:ymax, xmin:xmax, :] = crop_array

    
    

crop_map = Image.fromarray(bg_array)
crop_map.save(os.path.join(save_path, "0016.jpg"))
print(f"found {objects_count} objects in total {total_images_count} images")
