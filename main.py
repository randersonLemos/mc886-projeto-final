import os
import cv2
import time
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
    def __init__(self):
        network = IECore().read_network(
              'intel/face-detection-0202/FP32/face-detection-0202.xml'
            , 'intel/face-detection-0202/FP32/face-detection-0202.bin'
        )
        
        network = IECore().load_network(network)


if __name__ == '__main__':
    cap = cv2.VideoCapture('video001.mp4')
    while True:
        _, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(10)
