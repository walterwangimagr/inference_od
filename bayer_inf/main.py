import numpy as np
from detectors import SSD_OD
from PIL import Image, ImageDraw
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
from bayer_to_RGB import bayer_to_rgb
from pycoral.adapters import classify
import glob
import os
import re
import pprint

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
            id = obj.id
            detection = [xmin, ymin, xmax, ymax, score, id]
            detections.append(detection)

        return np.array(detections)
    else:
        return np.empty((0, 5))


bayer_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/sonae_test/bayer/testset"

model_path = "/home/walter/nas_cv/object_detection/tensorflow/models/tpu_models/mobilenetv2_full_basket/walter_exp/default_bayer/auk_mul_classes_2/export/auk_mul_2703.tflite"

dataset_name = os.path.basename(bayer_dir)
model_name = os.path.basename(model_path)
model_name = re.sub(".tflite", "", model_name)

save_path = f"/home/walter/nas_cv/walter_stuff/git/modular-end2end-testing/od_eval/result/{dataset_name}_{model_name}"
os.makedirs(save_path, exist_ok=True)


bayer_files = glob.glob(f"{bayer_dir}/*.bayer_8")

objects_count = 0
total_images_count = len(bayer_files)
for bayer_file in bayer_files:
    bayer_data = np.fromfile(bayer_file, np.uint8).reshape((1080, 1920, 1))
    img = bayer_to_rgb(bayer_file)
    # img.show()

    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    common.set_input(interpreter, bayer_data)
    interpreter.invoke()

    output_detail = interpreter.get_output_details()
    pprint.pprint(output_detail)
    value = interpreter.tensor(1)()
    pprint.pprint(value)
    break

    objs = detect.get_objects(
        interpreter,
        score_threshold=0.5,
        image_scale=(2, 2))
    

    detections = parse_objs(objs)
    
    for detection in detections:
        xmin = detection[0]
        ymin = detection[1]
        xmax = detection[2]
        ymax = detection[3]
        confidence = detection[4]
        id = detection[5]   
        if id == 0:
            ImageDraw.Draw(img).rectangle(
                [(xmin, ymin), (xmax, ymax)], outline='green')
        if id == 1:
            ImageDraw.Draw(img).rectangle(
                [(xmin, ymin), (xmax, ymax)], outline='red')
        if id == 2:
            ImageDraw.Draw(img).rectangle(
                [(xmin, ymin), (xmax, ymax)], outline='yellow')

    base_name = os.path.basename(bayer_file)
    save_name = re.sub(".bayer_8", ".jpg", base_name)
    img.save(os.path.join(save_path, save_name))
    if len(detections) > 0:
        objects_count += 1

print(f"found {objects_count} objects in total {total_images_count} images")
