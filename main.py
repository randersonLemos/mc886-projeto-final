import os
import cv2
import time
import copy
import queue
import numpy as np
import tensorflow as tf
from threading import Thread
from tensorflow.keras import layers
from tensorflow.keras import regularizers
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


class FaceEmotion:
    def __init__(self, path):

        self.model = tf.keras.models.load_model(path)

        self.class_names = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']    


    def emotion(self, face):
        face = cv2.resize(copy.copy(face), (224, 224), interpolation = cv2.INTER_AREA)

        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        face = np.expand_dims(face, 0)
        pred = self.model(face)
        return self.class_names[np.argmax(pred)]


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
    def draw_crop(cls, frame, crops):
        H, W, _ = frame.shape
        h = 10
        for crop in crops:
            ch, cw, _ = crop.shape
            if (h + ch) < H:
                frame[h:ch+h,0:cw] = crop
                h = h+1
            else:
                break
        return frame


    @classmethod
    def draw_text(cls, frame, pos, text):
        frame = copy.copy(frame)
        cv2.putText(frame, text, pos
                    , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255,), 3
                    , cv2.LINE_AA
                   )
        return frame



if __name__ == '__main__':
    fd = FaceDetector(confidence=0.5)
    fe = FaceEmotion('transferlearn_model_v1/model')
    fe.model.summary()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('video/output.mp4', fourcc, 15.0, (640,480))

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if ret:
            bboxs, confs = fd.bbox(frame)
            frame = Designer.draw_bbox(frame, bboxs)
            crops = Designer.crop_bbox(frame, bboxs)
            if crops:
                face = crops[0]
                emotion = fe.emotion(face)

                frame = Designer.draw_crop(frame, crops)
                frame = Designer.draw_text(frame, (10,30,), emotion) 

                out.write(frame)
                cv2.imshow('frame', frame)
                c = cv2.waitKey(1)
                if c & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
