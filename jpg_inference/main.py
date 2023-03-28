import numpy as np
from detectors import SSD_OD
from PIL import Image, ImageDraw
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
from bayer_to_RGB import bayer_to_rgb
from pycoral.adapters import classify
from data_util import padded_resize
import glob
import os
import re
import pprint
import platform


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
    

def make_save_path(src_dir, model_path):
    dataset_name = os.path.basename(src_dir)
    model_name = os.path.basename(model_path)
    model_name = re.sub(".tflite", "", model_name)
    cur_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
    result_path = os.path.join(parent_dir, "results")
    save_path = f"{result_path}/{dataset_name}_{model_name}"
    os.makedirs(save_path, exist_ok=True)
    return save_path







image_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/sonae_test/stack_bayer_white_balance/testset/"
model_path = "/home/walter/nas_cv/walter_stuff/git/yolov5-master/yolo_n_modular/yolo5_nano_448/weights/no_nms_edgetpu.tflite"


save_path = make_save_path(image_dir, model_path)

image_files = glob.glob(f"{image_dir}/*.jpg")
img_path = os.path.join(image_dir, "1668743387444-modular-coral-v1-akl-0101:169.254.247.0.jpg")

interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]
print(input_detail)
print(output_detail)


image = Image.open(img_path)
resized_image = padded_resize(image, 448, 448)
resized_image.show()
image_array = np.array(resized_image, dtype=np.uint8)

#de-scale
scale, zero_point = input_detail['quantization']
image_array = (image_array / scale - zero_point).astype(np.uint8)


common.set_input(interpreter, image_array)
interpreter.invoke()
y = interpreter.get_tensor(output_detail['index'])
scale, zero_point = output_detail['quantization']
y = (y.astype(np.float32) - zero_point) * scale
y[..., :4] *= [448, 448, 448, 448]
print(y)
    
    # output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
    
    # print(output.shape)
    # pprint.pprint(output_detail)
    # pprint.pprint(type(value))
    # objs = detect.get_objects(
    #     interpreter,
    #     score_threshold=0.5)
    
    # detections = parse_objs(objs)

    # for detection in detections:
    #     xmin = detection[0]
    #     ymin = detection[1]
    #     xmax = detection[2]
    #     ymax = detection[3]
    #     confidence = detection[4]
    #     id = detection[5]   
    #     if id == 0:
    #         ImageDraw.Draw(resized_image).rectangle(
    #             [(xmin, ymin), (xmax, ymax)], outline='green')
    #     if id == 1:
    #         ImageDraw.Draw(resized_image).rectangle(
    #             [(xmin, ymin), (xmax, ymax)], outline='red')
    #     if id == 2:
    #         ImageDraw.Draw(resized_image).rectangle(
    #             [(xmin, ymin), (xmax, ymax)], outline='yellow')
    
    # resized_image.show()
    

# objects_count = 0
# total_images_count = len(bayer_files)
# for bayer_file in bayer_files:
#     bayer_data = np.fromfile(bayer_file, np.uint8).reshape((1080, 1920, 1))
#     img = bayer_to_rgb(bayer_file)
#     # img.show()

#     interpreter = edgetpu.make_interpreter(model_path)
#     interpreter.allocate_tensors()
#     common.set_input(interpreter, bayer_data)
#     interpreter.invoke()
#     objs = detect.get_objects(
#         interpreter,
#         score_threshold=0.5,
#         image_scale=(2, 2))
    

#     detections = parse_objs(objs)
    
#     for detection in detections:
#         xmin = detection[0]
#         ymin = detection[1]
#         xmax = detection[2]
#         ymax = detection[3]
#         confidence = detection[4]
#         id = detection[5]   
#         if id == 0:
#             ImageDraw.Draw(img).rectangle(
#                 [(xmin, ymin), (xmax, ymax)], outline='green')
#         if id == 1:
#             ImageDraw.Draw(img).rectangle(
#                 [(xmin, ymin), (xmax, ymax)], outline='red')
#         if id == 2:
#             ImageDraw.Draw(img).rectangle(
#                 [(xmin, ymin), (xmax, ymax)], outline='yellow')

#     base_name = os.path.basename(bayer_file)
#     save_name = re.sub(".bayer_8", ".jpg", base_name)
#     img.save(os.path.join(save_path, save_name))
#     if len(detections) > 0:
#         objects_count += 1

# print(f"found {objects_count} objects in total {total_images_count} images")
