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


def decode(encoded_box, anchor_box):
    """
    input: encoded_box [ty,tx,th,tw]
    input: anchor_box [ycenter_a,xcenter_a,ha,wa]
    return: decoded_box [ymin,xmin,ymax,xmax]
    """
    ty = encoded_box[0]
    tx = encoded_box[1]
    th = encoded_box[2]
    tw = encoded_box[3]

    ycenter_a = anchor_box[0]
    xcenter_a = anchor_box[1]
    ha = anchor_box[2]
    wa = anchor_box[3]

    # scale_factors = BOX_CODER_SCALE
    scale_factors = [10.0, 10.0, 5.0, 5.0]

    ty /= scale_factors[0]
    tx /= scale_factors[1]
    th /= scale_factors[2]
    tw /= scale_factors[3]

    print("tw", tw)
    print("th", th)

    w = np.exp(tw) * wa
    h = np.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a

    

    ymin = max(0., ycenter - h / 2.)
    xmin = max(0., xcenter - w / 2.)
    ymax = min(1., ycenter + h / 2.)
    xmax = min(1., xcenter + w / 2.)

    return [ymin, xmin, ymax, xmax]


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
            return np.empty((0,5))


bayer_dir = "/home/walter/big_daddy/walter_stuff/sesame_bar"
bayer_files = glob.glob(f"{bayer_dir}/*0016*.bayer_8")
# bayer_files = ["/home/walter/Downloads/end_img.bayer_8"]
model_path = "/home/walter/nas_cv/object_detection/tensorflow/models/tpu_models/mobilenetv2_full_basket/walter_exp/sep21_cloth/exported/sep21_nms_edgetpu.tflite"
model_path = "/home/walter/nas_cv/object_detection/tensorflow/models/tpu_models/mobilenetv2_full_basket/walter_exp/default_bayer/exp20/export/model_edgetpu.tflite"
# save_path = "/home/walter/nas_cv/walter_stuff/save_result/productions_sesame"
save_path = "/home/walter/Downloads/raw_output"
os.makedirs(save_path, exist_ok=True)


for bayer_file in bayer_files:
    bayer_data = np.fromfile(bayer_file, np.uint8).reshape((1080, 1920, 1))
    img = bayer_to_rgb(bayer_file)
    # img.show()

    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    common.set_input(interpreter, bayer_data)
    interpreter.invoke()

    encoded_bboxs = common.output_tensor(interpreter, 0)[0]  # (3336,4)
    class_predictions = common.output_tensor(interpreter, 1)[0]  # (3336,2)
    anchors = common.output_tensor(interpreter, 2)  # (3336,4)
    scores = np.squeeze(class_predictions[..., 1] / 255)
    idx = np.where(scores > 0.4)
    s_encoded_bboxs = encoded_bboxs[idx] / 255
    scores = scores[idx]
    s_anchors = anchors[idx] / 255

    detections = []
    for i in range(len(scores)):
        bbox = decode(s_encoded_bboxs[i], s_anchors[i]) #[ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = bbox 
        detection = [xmin, ymin, xmax, ymax, scores[i]]
        detections.append(detection)
    


    for detection in detections:
        xmin = detection[0] * 640
        ymin = detection[1] * 360
        xmax = detection[2] * 640
        ymax = detection[3] * 360
        confidence = detection[4]
        ImageDraw.Draw(img).rectangle([(xmin, ymin), (xmax, ymax)], outline='green')
        ImageDraw.Draw(img).text(xy=(xmin,ymin-10), text='%.2f' % (confidence), fill='red')


    base_name = os.path.basename(bayer_file)
    save_name = re.sub(".bayer_8", ".jpg", base_name)
    img.save(os.path.join(save_path, save_name))
    
    

