import numpy as np
from detectors import SSD_OD
import cv2
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
from bayer_to_RGB import bayer_to_rgb
from pycoral.adapters import classify
from utils import pad_image, non_max_suppression
import glob
import os
import re
import torch



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


def load_image(img_path, img_size=448):
    '''
    Load Image and preprocessing, resize, pad, cvt color

    return im: np_array, im_to_draw: cv2 img
    '''
    im = cv2.imread(img_path)

    # original height and width
    o_height, o_width = im.shape[:2]
    # the edgetpu model require image to be a specify size
    r_im_size = img_size
    ratio = r_im_size / max(o_height, o_width)
    # resize image
    if ratio != 1:
        # use INTER_LINEAR to scale up and use INTER_AREA to scale down
        interp = cv2.INTER_LINEAR if (ratio > 1) else cv2.INTER_AREA
        im = cv2.resize(im, (int(o_width * ratio),
                        int(o_height * ratio)), interpolation=interp)

    im, ratio, pad = pad_image(im, (img_size, img_size), auto=False, scaleup=False)
    # to draw bbox and show
    im_to_draw = im


    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # turn cv2 image to np array and normalized
    im_array = np.asarray(im, dtype=np.float32)
    im_array /= 255
    im_array = np.expand_dims(im_array, axis=0)
    im_array = np.ascontiguousarray(im_array)

    return im, im_to_draw



image_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/sonae_test/stack_bayer_white_balance/testset"
model_path = "/home/walter/nas_cv/walter_stuff/git/yolov5-master/yolo_n_modular/yolo5_nano_448/weights/no_nms_edgetpu.tflite"


save_dir = make_save_path(image_dir, model_path)

image_files = glob.glob(f"{image_dir}/*.jpg")



'''Inference'''
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

for img_path in image_files:
    im_array, im_to_draw = load_image(img_path)
    # de-scale
    scale, zero_point = input_detail['quantization']
    im_array = (im_array / scale - zero_point).astype(np.uint8)

    # inference
    common.set_input(interpreter, im_array)
    interpreter.invoke()
    y = interpreter.get_tensor(output_detail['index'])

    # de-scale
    scale, zero_point = output_detail['quantization']
    y = (y.astype(np.float32) - zero_point) * scale
    # scale up to [w, h, w, h]
    y[..., :4] *= [448, 448, 448, 448]
    y = torch.tensor(y, device='cpu')
    out = non_max_suppression(y, 0.5, 0.5,  multi_label=True, agnostic=False)
    # out_np = out.numpy()
    detections = out[0].numpy()

    # draw bbox
    for detection in detections:
        xmin = int(detection[0])
        ymin = int(detection[1])
        xmax = int(detection[2])
        ymax = int(detection[3])
        confidence = detection[4]
        id = detection[5]
        
        if id == 0:
            cv2.rectangle(im_to_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if id == 1:
            cv2.rectangle(im_to_draw, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        if id == 2:
            cv2.rectangle(im_to_draw, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    # save image to dir 
    img_basename = os.path.basename(img_path)
    save_path = os.path.join(save_dir, img_basename)
    cv2.imshow("asd", im_to_draw)
    cv2.waitKey(0)
    break
    cv2.imwrite(save_path, im_to_draw)



