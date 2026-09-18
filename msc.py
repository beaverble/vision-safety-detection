import os
import sys
import signal
import argparse
import signal
import threading
#from threading import Thread, Event, interrupt_all_threads
import time
import zmq
from cbor2 import loads, dumps
import zstd
import numpy as np
import json
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

internal_cmd_sock = None

def handler_stop_signals(signum, frame):
    global internal_cmd_sock
    if internal_cmd_sock is None:
        sys.exit(0)
    else:
       internal_cmd_sock.send(b'stop')


#signal.signal(signal.SIGINT, handler_stop_signals)
signal.signal(signal.SIGTERM, handler_stop_signals)

#weights='./best.pt'
weights='best.pt'
device = torch.device('cuda')
half = True
dnn=False
#data = './data/coco.yaml'
data = 'data/coco.yaml'
imgsz=(640, 640)
augment = False
visualize = False
conf_thres=0.25
iou_thres=0.45
classes=None
agnostic_nms=False
max_det=100

NM_HEALTH_ADDR = 'tcp://HOST:PORT'
NM_COMMANDER_ADDR = 'tcp://HOST:PORT'
IN_ADDR = 'tcp://HOST:PORT'
OUT_ADDR = 'tcp://HOST:PORT'
INTERNAL_CMD_ADDR = 'inproc://incmd'

parser = argparse.ArgumentParser()
parser.add_argument('id')

args = parser.parse_args()

class ServiceExit(Exception):
    pass

class HealthReportJob(threading.Thread):

    def __init__(self, reportId, sock):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.reportId = reportId
        self.sock = sock

    def run(self):
        report_data = self.reportId + " " + str(os.getpid())
        while not self.shutdown_flag.is_set():
            try:
                self.sock.send_string(report_data)
                time.sleep(1)
            finally:
                pass


class InferenceJob(threading.Thread):

    def __init__(self, insock, outsock, controlsock, internal_cmd_sock, model_id):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.insock = insock
        self.outsock = outsock
        self.controlsock = controlsock
        self.internal_cmd_sock = internal_cmd_sock
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.model_id = model_id

    def run(self):
        print('Infer Thread #%s started' % self.ident)
        poller = zmq.Poller()
        poller.register(self.insock, zmq.POLLIN)
        poller.register(self.controlsock, zmq.POLLIN)

        while not self.shutdown_flag.is_set():
            try:
                socks = dict(poller.poll(500))
                if self.insock in socks and socks[self.insock] == zmq.POLLIN:
                    cbor_stream = self.insock.recv()
                    cbor_obj = loads(cbor_stream)
                    source_name = cbor_obj[0]
                    epochtime = cbor_obj[1]
                    millis = cbor_obj[2]
                    rows = cbor_obj[3]
                    cols = cbor_obj[4]
                    pixel_size = cbor_obj[5]
                    compress_image = cbor_obj[6]
                    intime = time.time()
                    #print(f'{epochtime} {millis} {intime}')
                    brg_data = zstd.decompress(compress_image)
                    mat = np.frombuffer(brg_data, dtype=np.uint8)
                    #print(self.model_id, source_name, epochtime, millis, len(compress_image))

                    img = mat.reshape(-1, cols, pixel_size)

                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = self.model(img, augment=augment, visualize=visualize)

                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    total_result = []
                    for i, det in enumerate(pred):
                        s, im0 = '', mat
                        if len(det):
                            total_result = []
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()
                                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                            for *xyxy, conf, cls in reversed(det):
                                class_num = int(cls)
                                if class_num < 100:
                                    class_num = class_num + 101
                                class_num = class_num
                                confidence = np.round(float(conf), 2)
                                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                                now_result = {
                                    "classType": class_num,
                                    "confidence" : confidence,
                                    "box": box
                                }
                                total_result.append(now_result)
                    detect_result = {
                        "result": total_result
                    }
                    converted_result = json.dumps(detect_result)
                    restime = time.time()
                    #print(f'{epochtime} {millis} {intime} {restime}')
                    self.outsock.send(dumps([source_name,epochtime,millis,rows,cols,pixel_size,compress_image,converted_result]))

                if self.controlsock in socks and socks[self.controlsock] == zmq.POLLIN:
                    command = self.controlsock.recv_string()
                    print("nc command %s", command)
                    self.internal_cmd_sock.send(b'stop')
                    #    os.kill(os.getpid(), signal.SIGINT)
                    # stop                 os.kill(os.getpid(), signal.SIGINT)
                    # mtp up down
            except Exception as e:
                print(e)
                pass
            finally:
                pass
        # ... Clean shutdown code here ...
        print('Thread #%s stopped' % self.ident)

zcontext = zmq.Context.instance()

def service_shutdown(signum, frame):
    print('Caught signal %d' % signum)
    raise ServiceExit

signal.signal(signal.SIGTERM, service_shutdown)
# signal.signal(signal.SIGINT, service_shutdown)

def main(argv, args):
    global internal_cmd_sock

    internal_cmd_sock = zcontext.socket(zmq.PUB)
    internal_cmd_sock.bind(INTERNAL_CMD_ADDR)

    print("main")
    stream_in_sock = zcontext.socket(zmq.PULL)
    stream_in_sock.set(zmq.SocketOption.CONFLATE, True)
    stream_in_sock.connect(IN_ADDR)
    stream_out_sock = zcontext.socket(zmq.PUSH)
    stream_out_sock.set(zmq.SocketOption.CONFLATE, True)
    stream_out_sock.connect(OUT_ADDR)

    health_report_sock = zcontext.socket(zmq.DEALER)
    health_report_sock.set(zmq.SocketOption.CONFLATE, True)
    health_report_sock.connect(NM_HEALTH_ADDR)

    commander_sock = zcontext.socket(zmq.DEALER)
    commander_sock.setsockopt_string(zmq.SocketOption.ROUTING_ID, args.id)
    commander_sock.connect(NM_COMMANDER_ADDR)

    healthReportJob = HealthReportJob(args.id, health_report_sock)
    inferenceJob = InferenceJob(stream_in_sock, stream_out_sock, commander_sock, internal_cmd_sock, args.id)
    healthReportJob.daemon = True
    inferenceJob.daemon = True
    healthReportJob.start()
    inferenceJob.start()

    internal_cmd_read_sock = zcontext.socket(zmq.SUB)
    internal_cmd_read_sock.set(zmq.SocketOption.SUBSCRIBE, b'')
    internal_cmd_read_sock.connect(INTERNAL_CMD_ADDR)

    stop_message = internal_cmd_read_sock.recv_string()

    healthReportJob.shutdown_flag.set()
    inferenceJob.shutdown_flag.set()
    healthReportJob.join()
    inferenceJob.join()

if __name__ == '__main__' :
    argv = sys.argv
    main(args,args)
