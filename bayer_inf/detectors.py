from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
import numpy as np



class SSD_OD():

    def __init__(self, model_path, model_confidence_threshold):
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.confidence = model_confidence_threshold
        self.interpreter.allocate_tensors()
        self.scale = (2,2)


    def predict(self, bayer_file):
        bayer_data = np.fromfile(bayer_file, np.uint8).reshape((1080, 1920, 1))
        common.set_input(self.interpreter, bayer_data)
        self.interpreter.invoke()
        objs = detect.get_objects(
                self.interpreter, 
                score_threshold=self.confidence, 
                image_scale=self.scale)

        detections = self.parse_objs(objs)
        return detections

    def parse_objs(self, objs):
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