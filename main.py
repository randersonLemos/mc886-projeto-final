import os
import cv2
import time
import copy
import queue
from threading import Thread
from openvino.inference_engine import IECore


class StreamReader(Thread):
    def __init__(self, address):
        super().__init__()

        self.address = 0 if address == '0' else address
        self.cap= cv2.VideoCapture(address)

        self.q = queue.Queue()

    def run(self): # Read frames as soon as they are available, keeping only most recent one
        while True:
            ret, frame = self.cap.read()

            if not ret:  # lost connection, must retry
                self.cap.release()
                time.sleep(3)
                self.cap = cv2.VideoCapture(self.address)
                continue

            try:
                self.q.get_nowait()  # Discard previous (unprocessed) frame
            except queue.Empty:
                pass

            self.q.put(frame)

    def read(self):
        return self.q.get()


class FaceDetector:
    def __init__(self, confidence):
        self.confidence = confidence


        network = IECore().read_network(
              'intel/face-detection-0202/FP32/face-detection-0202.xml'
            , 'intel/face-detection-0202/FP32/face-detection-0202.bin'
        )
        
        self.network = IECore().load_network(network)

        self.input_layer_name   = next( iter(self.network.input_info) )
        self.output_layer_name  = next( iter(self.network.outputs) )
        self.tensor_description = self.network.input_info[self.input_layer_name].tensor_desc


    def detect(self, mat):
        N, C, H, W = self.tensor_description.dims
        hwc = cv2.resize(mat, (W, H,))
        chw = hwc.transpose((2, 0, 1))
        res = self.network.infer({self.input_layer_name : chw})
        out = res[self.output_layer_name]
        return out


    def bbox(self, mat):
       out = self.detect(mat)
       H, W, _ = mat.shape
       bboxs = []
       confs = []

       for detection in out[0, 0, :, :]:
            idd, label, conf, xmin, ymin, xmax, ymax = detection
            if conf > self.confidence:
                xmin = max(int(xmin * W), 0)
                ymin = max(int(ymin * H), 0)
                xmax = min(int(xmax * W), W)
                ymax = min(int(ymax * H), H)

                x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin

                bbox = (x, y, w, h)

                bboxs.append(bbox); confs.append(conf)

       return bboxs, confs


class Designer:
    @classmethod
    def draw_bbox(cls, frame, bboxs):
        frame = copy.copy(frame)
        for bbox in bboxs:
            x, y, w, h = bbox
            frame = cv2.rectangle(frame, (x, y, ), (x+w, y+h, ), (255, 0, 0, ), 2)
        return frame


    @classmethod
    def crop_bbox(cls, frame, bboxs):
        crops = []
        for bbox in bboxs:
            x, y, w, h = bbox
            crop = frame[y:y+h, x:x+w]
            crops.append(crop)
        return crops


    @classmethod
    def add_crop(cls, frame, crops):
        H, W, _ = frame.shape
        h = 0
        for crop in crops:
            ch, cw, _ = crop.shape
            if (h + ch) < H:
                frame[h:ch,0:cw] = crop
                b = h+1
            else:
                break
        return frame


if __name__ == '__main__':
    fd = FaceDetector(confidence=0.5)

    #cap = cv2.VideoCapture('video001.mp4')
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        bboxs, confs = fd.bbox(frame)
        frame = Designer.draw_bbox(frame, bboxs)
        crops = Designer.crop_bbox(frame, bboxs)
        frame = Designer.add_crop(frame, crops)
        cv2.imshow('frame', frame)
        cv2.waitKey(2)

        
