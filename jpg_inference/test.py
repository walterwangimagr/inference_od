import os 
import re 

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


bayer_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/sonae_test/bayer/testset"
image_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/sonae_test/stack_bayer_white_balance/testset"

model_path = "/home/walter/nas_cv/walter_stuff/git/yolov5-master/yolo_n_modular/yolo5_nano_448/weights/no_nms_edgetpu.tflite"
print(make_save_path(image_dir,model_path))
