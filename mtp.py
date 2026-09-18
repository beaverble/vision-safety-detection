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
import math
from collections import Counter
import base64
import logging
from datetime import timedelta
from datetime import datetime
from utils.augmentations import letterbox
from utils.general import scale_boxes
import cv2

internal_cmd_sock = None
weights='./target_1280.pt'
device = torch.device('cuda')
half = True
dnn=False
data = './data/coco.yaml'
imgsz=(640, 640)
augment = False
visualize = False
conf_thres=0.25
iou_thres=0.45
classes=None
agnostic_nms=False
max_det=100

start_time = datetime.now().strftime('%Y%m%d%H%M%S')
logger_file = logging.getLogger()
logger_file.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
log_info_name = './mtplog/mtp_' + start_time + '.log'
file_handler = logging.FileHandler(log_info_name, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger_file.addHandler(file_handler)

def handler_stop_signals(signum, frame):
    global internal_cmd_sock
    if internal_cmd_sock is None:
        sys.exit(0)
    else:
       internal_cmd_sock.send(b'stop')


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def find_line_equation(point1, point2):
    # Calculate the slope (m) and y-intercept (b) of the line passing through point1 and point2
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2:
        return None, x1  # Vertical line case
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def find_intersection(m1, b1, m2, b2, x_vertical=None):
    if m1 is None:
        # First line is vertical
        x = x_vertical
        y = m2 * x + b2
    elif m2 is None:
        # Second line is vertical
        x = x_vertical
        y = m1 * x + b1
    else:
        # Both lines are non-vertical
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    return x, y

def find_contours(img):
    # 엣지 검출
    edges = cv2.Canny(img, 25, 100)
    # 컨투어 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 컨투어 선택
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - bbox1, bbox2: Two bounding boxes in the format [x1, y1, x2, y2].

    Returns:
    - iou: IoU value.
    """
    # Extract coordinates for bbox1
    x1_1, y1_1, x2_1, y2_1 = bbox1
    # Extract coordinates for bbox2
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate the (x, y) coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calculate the area of both the bounding boxes
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate the area of union
    union_area = bbox1_area + bbox2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area

    return iou

def move_bbox(bbox, dx, dy):
    x_min, y_min, x_max, y_max = bbox
    return (x_min + dx, y_min + dy, x_max + dx, y_max + dy)

def get_best_bbox_direction(bboxes_old, bboxes_new, step_size=5):
    directions = [(0, 0), (step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                  (step_size, step_size), (step_size, -step_size), (-step_size, step_size), (-step_size, -step_size),
                  (step_size*2, 0), (-step_size*2, 0), (0, step_size*2), (0, -step_size*2),
                  (step_size*2, step_size*2), (step_size*2, -step_size*2), (-step_size*2, step_size*2), (-step_size*2, -step_size*2)]

    best_iou_direction = [0, 0]
    best_iou_sum = 0
    for dx, dy in directions:
        iou_sum = 0
        for new_bbox in bboxes_new:
            moved_bbox = move_bbox(new_bbox, dx, dy)
            best_iou = 0
            for old_bbox in bboxes_old:
                iou = calculate_iou(moved_bbox, old_bbox)
                if iou > best_iou:
                    best_iou = iou
            iou_sum = iou_sum + best_iou
        if iou_sum > best_iou_sum:
            best_iou_sum = iou_sum
            best_iou_direction[0] = dx
            best_iou_direction[1] = dy
    dx = best_iou_direction[0]
    dy = best_iou_direction[1]
    bboxes_new_copy = bboxes_new.copy()
    for i in range(len(bboxes_new_copy)):
        bboxes_new_copy[i] = move_bbox(bboxes_new_copy[i], dx, dy)
    return find_new_bboxes(bboxes_old, bboxes_new_copy, dx, dy)

def find_new_bboxes(bboxes_old, bboxes_new, dx, dy, iou_threshold=0.2):
    new_bboxes = []
    for new_bbox in bboxes_new:
        matched = False
        for old_bbox in bboxes_old:
            iou = calculate_iou(new_bbox, old_bbox)
            if iou > iou_threshold:
                matched = True
                break
        if not matched:
            new_bbox_old = move_bbox(new_bbox, -dx, -dy)
            new_bboxes.append(new_bbox_old)
    return new_bboxes

def calculate_distance_from_center(shape, bbox):
    # Get image dimensions
    height, width, _ = shape

    # Calculate image center
    image_center_x = width / 2
    image_center_y = height / 2

    # Get bbox coordinates
    x1, y1, x2, y2 = bbox

    # Calculate bbox center
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2

    # Calculate distance between bbox center and image center
    distance = math.sqrt((bbox_center_x - image_center_x) ** 2 + (bbox_center_y - image_center_y) ** 2)

    return distance

def find_tank(contour, img, shape):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # 추출된 모서리 출력
    data = approx.reshape(-1,2)
    sorted_data = data[data[:, 1].argsort()[::-1]]

    y1 = sorted_data[:4]
    y2 = sorted_data[4:6]
    y3 = sorted_data[6:-2]
    y4 = sorted_data[-2:]

    y1 = y1[y1[:, 0].argsort()]
    y2 = y2[y2[:, 0].argsort()]
    y3 = y3[y3[:, 0].argsort()]
    y4 = y4[y4[:, 0].argsort()]

    left = [y1[0], y3[0]]
    right = [y1[3], y3[-1]]
    top = y4

    p1, p2 = left
    q1, q2 = top

    m1, b1 = find_line_equation(p1, p2)
    m2, b2 = find_line_equation(q1, q2)

    if m1 is None or m2 is None:
        if m1 is None:
            left_top = find_intersection(m1, b1, m2, b2, x_vertical=p1[0])
        else:
            left_top = find_intersection(m1, b1, m2, b2, x_vertical=q1[0])
    else:
        left_top = find_intersection(m1, b1, m2, b2)

    p1, p2 = right

    m1, b1 = find_line_equation(p1, p2)
    m2, b2 = find_line_equation(q1, q2)

    if m1 is None or m2 is None:
        if m1 is None:
            right_top = find_intersection(m1, b1, m2, b2, x_vertical=p1[0])
        else:
            right_top = find_intersection(m1, b1, m2, b2, x_vertical=q1[0])
    else:
        right_top = find_intersection(m1, b1, m2, b2)

    left_top = tuple(int(x) for x in left_top)
    right_top = tuple(int(x) for x in right_top)
    left_bottom = y1[0]
    right_bottom = y1[3]
    src_pts = np.array([np.array(left_top), np.array(right_top), left_bottom, right_bottom], dtype="float32")
    dst_pts = np.array([[0, 0], [shape[1], 0], [0, shape[0]], [shape[1], shape[0]]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    width_ratio = shape[1] / 3350
    height_ratio = shape[0] / 2200

    new_left_top = np.array([[-1*(width_ratio*300), -1*(height_ratio*300)]], dtype=np.float32)
    new_right_top = np.array([[shape[1]+(width_ratio*300), -1*(height_ratio*300)]], dtype=np.float32)
    new_left_bottom = np.array([[-1*(width_ratio*300), shape[0]+(height_ratio*300)]], dtype=np.float32)
    new_right_bottom = np.array([[shape[1]+(width_ratio*300), shape[0]+(height_ratio*300)]], dtype=np.float32)
    new_total = np.array([new_left_top, new_right_top, new_left_bottom, new_right_bottom])
    ret, inverse_matrix = cv2.invert(M)

    transformed_points = cv2.perspectiveTransform(new_total.reshape(1,4,2),inverse_matrix)[0]

    new_src_pts = np.array(transformed_points.astype(int), dtype="float32")

    dst_pts = np.array([[0, 0], [int(shape[1]*0.8), 0], [0, shape[0]], [int(shape[1]*0.8), shape[0]]], dtype="float32")
    new_M = cv2.getPerspectiveTransform(new_src_pts, dst_pts)

    # 이미지 변환
    warped = cv2.warpPerspective(img, new_M, (int(shape[1]*0.8), shape[0]))
    return warped

def warped_tank(img, shape, point):
    # dst_pts = np.array([[0, 0], [shape[1], 0], [0, shape[0]], [shape[1], shape[0]]], dtype="float32")
    dst_pts = np.array([[0, 0], [int(shape[1]*0.8), 0], [0, shape[0]], [int(shape[1]*0.8), shape[0]]], dtype="float32")
    new_M = cv2.getPerspectiveTransform(point, dst_pts)
    # 이미지 변환
    warped = cv2.warpPerspective(img, new_M, (int(shape[1]*0.8), shape[0]))
    return warped

def pred_img(img_orig, shape_orig, model):
    img_size = 640
    im = letterbox(img_orig, img_size, stride=model.stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    img = torch.from_numpy(im).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=augment, visualize=visualize)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    total_result = []
    for i, det in enumerate(pred):
        #        print(det)
        s = ''
        if len(det):
            total_result = []
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], shape_orig).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                s += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                class_num = class_num
                confidence = np.round(float(conf), 2)
                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                now_result = {
                    "classType": class_num,
                    "confidence" : confidence,
                    "box": box
                }
                total_result.append(now_result)
    return total_result

def crop_tank(img_orig, model):
    img_orig_shape = img_orig.shape
    result_orig = pred_img(img_orig, img_orig_shape, model)

    contour_orig = find_contours(img_orig)
    tank_find = False
    contour_find = False
    tank_box = []
    for detection in result_orig:
        class_type = detection['classType']
        model_box = detection['box']

        if class_type != 0:
            continue
        else:
            tank_box = detection['box']
        contour_find, tank_find = check_contour_tank(contour_orig, model_box, img_orig_shape)
        break
    if contour_find == True:
        img_tank = find_tank(contour_orig, img_orig, img_orig_shape)
    elif tank_find == True:
        tank_x_len = tank_box[2] - tank_box[0]
        tank_y_len = tank_box[3] - tank_box[1]
        new_box = [[int(tank_box[0]+tank_x_len*0.075), int(tank_box[1]-tank_y_len*0.075)],
                   [int(tank_box[2]-tank_x_len*0.075), int(tank_box[1]-tank_y_len*0.075)],
                   [int(tank_box[0]-tank_x_len*0.075), int(tank_box[3]+tank_y_len*0.15)],
                   [int(tank_box[2]+tank_x_len*0.075), int(tank_box[3]+tank_y_len*0.15)]]
        img_tank = warped_tank(img_orig, img_orig_shape, np.array(new_box, dtype="float32"))
    else:
        new_box = [[int(img_orig_shape[1]*0.2), int(img_orig_shape[0]*0.05)],
                   [int(img_orig_shape[1]*0.8), int(img_orig_shape[0]*0.05)],
                   [int(img_orig_shape[1]*0.1), int(img_orig_shape[0]*0.9)],
                   [int(img_orig_shape[1]*0.9), int(img_orig_shape[0]*0.9)]]
        img_tank = warped_tank(img_orig, img_orig_shape, np.array(new_box, dtype="float32"))
    return img_tank, result_orig, contour_find, tank_find

def check_contour_tank(contour_orig, model_box, img_orig_shape):
    tank_find = False
    contour_find = False
    x, y, w, h = cv2.boundingRect(contour_orig)
    x1, y1, x2, y2 = model_box
    largest_contour_bbox = [x, y, x + w, y + h]

    iou = calculate_iou(largest_contour_bbox, model_box)
    if iou < 0.5:
        model_box_center = calculate_distance_from_center(img_orig_shape, model_box)
        if model_box_center < img_orig_shape[1] / 10:
            if (x2-x1) > img_orig_shape[1] / 2:
                tank_find = True
    else:
        bbox_area = w * h
        contour_area = cv2.contourArea(contour_orig)
        area_ratio = contour_area / bbox_area
        if area_ratio > 0.5:
            contour_find = True
    return contour_find, tank_find

def is_point_in_rectangle(px, py, rect):
    """
    특정 좌표(px, py)가 주어진 사각형(rect) 내에 있는지 검사합니다.
    rect는 (x, y) 좌표의 리스트로 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 형식입니다.
    """
    # 점이 다각형 내부에 있는지 확인하는 함수
    def is_point_in_polygon(x, y, polygon):
        n = len(polygon)
        inside = False

        px, py = polygon[0]
        for i in range(n+1):
            qx, qy = polygon[i % n]
            if y > min(py, qy):
                if y <= max(py, qy):
                    if x <= max(px, qx):
                        if py != qy:
                            xints = (y - py) * (qx - px) / (qy - py) + px
                        if px == qx or x <= xints:
                            inside = not inside
            px, py = qx, qy

        return inside
    return is_point_in_polygon(px, py, rect)

def find_new_point(save_folder, fire_before, fire_after):
    crop_folder_list = os.listdir(save_folder)
    crop_folder_list.sort()
    crop_image_list = []
    crop_txt_list = []
    for file_name in crop_folder_list:
        if file_name.__contains__('.jpg'):
            crop_image_list.append(file_name)
        else:
            crop_txt_list.append(file_name)
    image_tank_after = cv2.imread(save_folder+'/'+crop_image_list[fire_after])
    with open(save_folder+'/'+crop_txt_list[fire_before], 'r') as file:
        label_tank_before = json.load(file)
    with open(save_folder+'/'+crop_txt_list[fire_after], 'r') as file:
        label_tank_after = json.load(file)

    bbox_before = []
    bbox_after = []
    bbox_after_tank = []
    for detection in label_tank_before:
        class_type = detection['classType']
        if class_type == 0:
            continue
        bbox_before.append(tuple(detection['box']))

    for detection in label_tank_after:
        class_type = detection['classType']
        if class_type == 0:
            bbox_after_tank.append(tuple(detection['box']))
            continue
        bbox_after.append(tuple(detection['box']))

    new_bboxes = get_best_bbox_direction(bbox_before, bbox_after)
    if len(new_bboxes) == 0:
        return False, [0, 0, 0, 0], [0, 0], 0, crop_image_list[fire_after]
    now_confidence = 0
    for detection in label_tank_after:
        if list(new_bboxes[0]) == detection['box']:
            now_confidence = detection['confidence']

    now_point = [0, 0]
    if len(new_bboxes) == 1:
        x1, y1, x2, y2 = new_bboxes[0]
        now_point[0] = int((x1+x2)/2)
        now_point[1] = int((y1+y2)/2)

    contour_after = find_contours(image_tank_after)
    image_tank_after_shape = image_tank_after.shape

    contour_after_check, tank_after_check = check_contour_tank(contour_after, bbox_after_tank[0], image_tank_after_shape)
    hit=False
    if contour_after_check == True:
        if cv2.pointPolygonTest(contour_after, tuple(now_point), False) >= 0:
            hit = True
        M = cv2.moments(contour_after)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    elif tank_after_check == True:
        x1, y1, x2, y2 = bbox_after_tank[0]
        cx = int((x1 + x2)/2)
        cy = int((y1 + y2)/2)
        if x1 <= now_point[0] <= x2 and y1 <= now_point[1] <= y2:
            tank_x = (x2-x1)/3350
            tank_y = (y2-y1)/2200
            rect_1 = [(x1, y1), (x1+int(tank_x*575), y1), (x1, y1+int(tank_y*820)), (x1+int(tank_x*380), y1+int(tank_y*820))]
            rect_2 = [(x2, y1), (x2-int(tank_x*575), y1), (x2, y1+int(tank_y*820)), (x2-int(tank_x*380), y1+int(tank_y*820))]
            rect_3 = [(x1+int(tank_x*600), y2), (x1+int(tank_x*600), y2-int(tank_y*540)),
                      (x2-int(tank_x*600), y2), (x2-int(tank_x*600), y2-int(tank_y*540))]
            rect_1_check = is_point_in_rectangle(now_point[0], now_point[1], rect_1)
            rect_2_check = is_point_in_rectangle(now_point[0], now_point[1], rect_2)
            rect_3_check = is_point_in_rectangle(now_point[0], now_point[1], rect_3)
            if rect_1_check or rect_2_check or rect_3_check:
                hit = False
            else:
                hit = True
    else:
        cx = int(image_tank_after_shape[1]/2)
        cy = int(image_tank_after_shape[0]/2)
        x1 = image_tank_after_shape[1] * 0.25
        x2 = image_tank_after_shape[1] * 0.75
        y1 = image_tank_after_shape[0] * 0.10
        y2 = image_tank_after_shape[0] * 0.60
        if x1 <= now_point[0] <= x2 and y1 <= now_point[1] <= y2:
            hit = True


    x_ratio = 3950 / image_tank_after_shape[1]
    y_ratio = 2800 / image_tank_after_shape[0]
    now_point[0] = (now_point[0] - cx) * x_ratio * 0.9 + 2275
    now_point[1] = (cy - now_point[1]) * y_ratio * 0.9 + 1700
    if hit and now_point[0] > 1200 and now_point[0] < 3350 and now_point[1] < 1140:
        now_point[1] = 1140

    return hit, new_bboxes, now_point, now_confidence, crop_image_list[fire_after]



signal.signal(signal.SIGINT, handler_stop_signals)
signal.signal(signal.SIGTERM, handler_stop_signals)

NM_HEALTH_ADDR = 'tcp://HOST:PORT'
NM_COMMANDER_ADDR = 'tcp://HOST:PORT'
#NM_COMMANDER_ADDR = 'tcp://HOST:PORT'
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

    def __init__(self, insock, outsock, controlsock, internal_cmd_sock):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.insock = insock
        self.outsock = outsock
        self.controlsock = controlsock
        self.internal_cmd_sock = internal_cmd_sock
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.timeout = 9999
        self.state = -1
        self.up_time = -1
        self.up_millis = -1
        self.hit = 'N'
        self.hit_x = 0
        self.hit_y = 0
        self.hit_box = [0, 0, 0, 0]
        self.hit_confidence = 0
        self.hit_time = -1
        self.hit_millis = -1
        self.point_list = []
        self.contour_list = []
        self.tank_list = []
        self.filename_list = []
        self.total_point_list = []
        self.point_before = -1
        self.fire_before = -1
        self.fire_after = -1
        self.result_send = -1
        self.orig_folder = './target_image'
        self.save_folder = './target_crop'
        createFolder(self.orig_folder)
        createFolder(self.save_folder)
        file_list = os.listdir(self.orig_folder)
        for file_name in file_list:
            os.remove(self.orig_folder+'/'+file_name)
        file_list = os.listdir(self.save_folder)
        for file_name in file_list:
            os.remove(self.save_folder+'/'+file_name)

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
                    #print(len(cbor_stream))
                    cbor_obj = loads(cbor_stream)
                    source_name = cbor_obj[0]
                    epochtime = cbor_obj[1]
                    millis = cbor_obj[2]
                    rows = cbor_obj[3]
                    cols = cbor_obj[4]
                    pixel_size = cbor_obj[5]
                    #logger_file.info(str(len(cbor_stream)))
                    if self.state == 8:
                        self.up_time = -1
                        self.up_millis = -1
                        self.hit = 'N'
                        self.hit_x = 0
                        self.hit_y = 0
                        self.hit_box = [0, 0, 0, 0]
                        self.hit_confidence = 0
                        self.hit_time = -1
                        self.hit_millis = -1
                        self.point_list = []
                        self.contour_list = []
                        self.tank_list = []
                        self.filename_list = []
                        self.total_point_list = []
                        self.point_before = -1
                        self.fire_before = -1
                        self.fire_after = -1
                        self.result_send = -1

                    if self.state == 2 and self.up_time == -1:
                        self.up_time = int(epochtime)
                        self.up_millis = millis

                    if self.state == 3:
                        self.result_send = 1
                    if self.state == 2 and self.up_time != -1 and int(epochtime) - self.up_time > self.timeout:
                        self.result_send = 1

                    if self.state == 2: 
                        compress_image = cbor_obj[6]
                        brg_data = zstd.decompress(compress_image)
                        mat = np.frombuffer(brg_data, dtype=np.uint8)
                        image_name = str(epochtime)+'_'+str(millis)

                        img_orig = mat.reshape(-1, cols, pixel_size)
                        img_tank, result, contour_find, tank_find = crop_tank(img_orig, self.model)
                        result_tank = pred_img(img_tank, img_tank.shape, self.model)
                        # orig_name = image_name+'.jpg'
                        # orig_txt_name = orig_name+'.txt'
                        crop_name = image_name+'.jpg'
                        crop_txt_name = image_name+'.txt'
                        # with open(orig_folder+'/'+orig_txt_name, 'w') as file:
                        #     json.dump(result, file)
                        with open(self.save_folder+'/'+crop_txt_name, 'w') as file:
                            json.dump(result_tank, file)
                        cv2.imwrite(self.save_folder+'/'+crop_name, img_tank)
                        cv2.imwrite(self.orig_folder+'/'+crop_name, img_orig)
                    
                        now_point_count = 0
                        for detection in result_tank:
                            class_type = detection['classType']
                            if class_type == 0:
                                continue
                            else:
                                now_point_count = now_point_count + 1
                        self.point_list.append(now_point_count)
                        self.contour_list.append(contour_find)
                        self.tank_list.append(tank_find)
                        self.filename_list.append(image_name)
                        #if len(self.point_list) == 20:
                        if len(self.point_list) > 4:
                            counter = Counter(self.point_list)
                            mode_data = counter.most_common(1)
                            mode_value = mode_data[0][0]
                            mode_frequency = mode_data[0][1]

                            file_index = -1
                            for point_index in range(len(self.point_list)):
                                if self.point_list[point_index] == mode_value and self.contour_list[point_index]:
                                    file_index = point_index
                            if file_index == -1:
                                for point_index in range(len(self.point_list)):
                                    if self.point_list[point_index] == mode_value and self.tank_list[point_index]:
                                        file_index = point_index
                            for point_index in range(len(self.point_list)):
                                if point_index == file_index:
                                    self.total_point_list.append(mode_value)
                                    continue
                                orig_name = self.filename_list[point_index]
                                # orig_txt_name = orig_name+'.txt'
                                crop_name = orig_name+'.jpg'
                                crop_txt_name = orig_name+'.txt'
                                # orig_name = orig_name+'.jpg'
                                # mtp 코드에선 사용
                                os.remove(self.orig_folder+'/'+crop_name)
                                # os.remove(orig_folder+'/'+orig_txt_name)
                                os.remove(self.save_folder+'/'+crop_name)
                                os.remove(self.save_folder+'/'+crop_txt_name)
                            self.tank_list = []
                            self.contour_list = []
                            self.filename_list = []
                            self.point_list = []

                        #logger_file.info(self.total_point_list)
                        if len(self.total_point_list) > 4 and self.point_before == -1:
                            counter = Counter(self.total_point_list)
                            mode_data = counter.most_common(1)
                            mode_value = mode_data[0][0]
                            mode_frequency = mode_data[0][1]
                            if mode_frequency > len(self.total_point_list):
                                self.point_before = mode_value
                                for fire_index in range(-5, 0):
                                    if mode_value == self.total_point_list[fire_index]:
                                        self.fire_before = len(self.total_point_list) + fire_index
                        elif len(self.total_point_list) > 2 and self.point_before == -1:
                            if self.total_point_list[-1] == self.total_point_list[-2] and self.total_point_list[-1] == self.total_point_list[-3]:
                                self.point_before = self.total_point_list[-1]
                                self.fire_before = len(self.total_point_list) - 3
                    
                        if self.point_before > -1:
                            if self.total_point_list[-1] == self.total_point_list[-2] and self.total_point_list[-1] == self.total_point_list[-3]:
                                if self.point_before+1 == self.total_point_list[-1]:
                                    self.fire_after = len(self.total_point_list) - 4
                            elif len(self.total_point_list) > 4:
                                counter = Counter(self.total_point_list[-5:])
                                mode_data = counter.most_common(1)
                                mode_value = mode_data[0][0]
                                mode_frequency = mode_data[0][1]
                                if mode_frequency > 2 and self.point_before+1 == mode_value:
                                    for fire_index in range(-5, 0):
                                        if mode_value == self.total_point_list[fire_index]:
                                            self.fire_after = len(self.total_point_list) + fire_index
                    
                        if self.fire_after > -1:
                            try:
                                fire_hit, fire_bbox, fire_point, now_confidence, fire_time = find_new_point(self.save_folder, self.fire_before, self.fire_after)
                            except:
                                fire_hit, fire_bbox, fire_point, now_confidence = False, [0, 0, 0, 0], [0, 0], 0
                                fire_time = os.listdir(self.save_folder)
                                if len(fire_time) > 0:
                                    fire_time = fire_time[-1].split('.')[0]
                                else:
                                    fire_time = str(self.up_time)+'_'+str(self.up_millis)
                            print(fire_hit, fire_bbox, fire_point, now_confidence, fire_time)
                            fire_time = fire_time.split('_')
                            if fire_hit:
                                self.hit = 'Y'
                            self.hit_x = fire_point[0]
                            self.hit_y = fire_point[1]
                            self.hit_box = fire_bbox
                            self.hit_confidence = now_confidence
                            self.hit_time = fire_time[0]
                            self.hit_millis = fire_time[1]
                            self.result_send = 1
                        

                    if self.result_send == 1:
                        # 이미지 저장/불러오기 작업 처리
                        send_result = []
                        #st_time = datetime.datetime.fromtimestamp(int(self.hit_time))
                        #up_time = datetime.datetime.fromtimestamp(self.up_time)
                        st_time = datetime.fromtimestamp(int(self.hit_time))
                        up_time = datetime.fromtimestamp(self.up_time)
                        # Format datetime object to string
                        st_time = st_time.strftime('%Y%m%d%H%M%S')
                        st_time = st_time + str(self.hit_millis).zfill(3)[:2]
                        #st_time = st_time + self.hit_millis.zfill(3)[:2]
                        up_time = up_time.strftime('%Y%m%d%H%M%S')
                        #up_time = up_time + self.up_millis.zfill(3)[:2]
                        up_time = up_time + str(self.up_millis).zfill(3)[:2]
                        image_name = str(self.hit_time) + '_' + str(self.hit_millis) + '.jpg'
                        try:
                            send_orig_image = zstd.compress(cv2.imread(self.orig_folder+'/'+image_name).reshape(-1), 1)
                            with open(self.save_folder+'/'+image_name, "rb") as jpg_file:
                                send_crop_image = base64.b64encode(jpg_file.read())
                                send_crop_image = send_crop_image.decode('utf-8')
                        except:
                            send_file_list = os.listdir(self.orig_folder)
                            send_crop_list = os.listdir(self.save_folder)
                            if len(send_file_list) > 0:
                                send_orig_image = zstd.compress(cv2.imread(self.orig_folder+'/'+send_file_list[0].split('.')[0]+'.jpg').reshape(-1), 1)
                            else:
                                send_orig_image = '0'
                            if len(send_crop_list) > 0:
                                with open(self.save_folder+'/'+send_crop_list[0].split('.')[0]+'.jpg', "rb") as jpg_file:
                                    send_crop_image = base64.b64encode(jpg_file.read())
                                    send_crop_image = send_crop_image.decode('utf-8')
                            else:
                                send_crop_image = '0'
                        now_result = {
                            "classType": 201,
                            "confidence" : self.hit_confidence,
                            "box": self.hit_box,
                            "hit": {
                                "st_time": st_time,
                                "up_time": up_time,
                                "hit": self.hit,
                                "sp_time": self.up_time - epochtime,
                                "hit_x": self.hit_x,
                                "hit_y": self.hit_y,
                                "crop_img": send_crop_image
                            }
                        }
                        send_result.append(now_result)
                        detect_result = {
                            "result": [now_result]
                        }
                        converted_result = json.dumps(detect_result)
                        self.outsock.send(dumps([source_name,self.hit_time,self.hit_millis,rows,cols,pixel_size,send_orig_image,converted_result]))
                        file_list = os.listdir(self.orig_folder)
                        for file_name in file_list:
                            os.remove(self.orig_folder+'/'+file_name)
                        file_list = os.listdir(self.save_folder)
                        for file_name in file_list:
                            os.remove(self.save_folder+'/'+file_name)
                if self.controlsock in socks and socks[self.controlsock] == zmq.POLLIN:
                    command = self.controlsock.recv_string()
                    print("nc command ", command)
                    if command.startswith('stop'):
                        self.internal_cmd_sock.send(b'stop')
                    elif command.startswith('post state'):
                        self.state = int(command.split(' ')[2])
                    elif command.startswith('post timeout'):
                        self.timeout = int(command.split(' ')[2])
                    logger_file.info('state:'+str(self.state)+' timeout:'+str(self.timeout))
                    #    os.kill(os.getpid(), signal.SIGINT)
                    # stop                 os.kill(os.getpid(), signal.SIGINT)
                    # mtp up down
            #except Exception as e:
            #    print(e)
            #    pass
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
    logger_file.info('id:'+str(args.id))
    healthReportJob = HealthReportJob(args.id, health_report_sock)
    inferenceJob = InferenceJob(stream_in_sock, stream_out_sock, commander_sock, internal_cmd_sock)
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
