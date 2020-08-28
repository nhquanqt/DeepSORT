import cv2
import numpy as np
import json
import random

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

label_colors = {
    -1: (127,127,127),
    1 : (0,0,0),
    2 : (255,0,0),
    3 : (0,255,0),
    4 : (0,0,255),
}

threshold = {
    1 : 0.1,
    2 : 0.4,
    3 : 0.4,
    4 : 0.4
}

cam_num = 'cam_02'

f = open('datasets/{}.txt'.format(cam_num))
events = f.readlines()

f = json.load(open('datasets/{}.json'.format(cam_num)))
zone = f['shapes'][0]['points']
zone = np.array(zone, np.int32)

# Definition of the parameters
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0

#initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

tracker = Tracker(metric)

track_labels = {}

def find_tracking_label(tracking_box, boxes, iou_thresh = 0.3):
    '''
    Args:
        tracking_box: shape (4) and format [x_min, y_min, x_max, y_max]
        boxes: detection with shape (num, 5) and format [x1, y1, x2, y2, label]
    Returns:
        label of the tracking box
    '''
    if len(boxes) == 0:
        return -1
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    x_min, y_min, x_max, y_max = tracking_box
    tracking_area = (x_max - x_min + 1) * (y_max - y_min + 1)

    xx1 = np.maximum(x_min, x1)
    yy1 = np.maximum(y_min, y1)
    xx2 = np.minimum(x_max, x2)
    yy2 = np.minimum(y_max, y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    intersection = w * h

    iou = intersection / (areas + tracking_area - intersection)
    
    if np.max(iou) < iou_thresh:
        return -1
    return boxes[np.argmax(iou), -1]

def hard_nms(boxes, label, scores, iou_thresh=None):
    """The basic hard non-maximum suppression.
    Args:
        dets: detection with shape (num, 5) and format [x1, y1, x2, y2, score].
        iou_thresh: IOU threshold,
    Returns:
        numpy.array: Retained boxes.
    """
    iou_thresh = iou_thresh or 0.5
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(overlap <= iou_thresh)[0]
        order = order[inds + 1]

    return boxes[keep], label[keep], scores[keep]

frame_id = 0
event_id = 0

track_dict = {}
track_colors = {}

cap = cv2.VideoCapture('datasets/{}.mp4'.format(cam_num))
while cap.isOpened():
    ret, frame = cap.read()

    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = np.array(image, np.float32) / 255.

    frame = cv2.polylines(frame, [zone], True, (127,127,127), 5)

    bboxes = []
    confidents = []
    labels = []
    dets = []

    try:
        while True:
            e = list(map(float, events[event_id].split(' ')))
            if e[0] != frame_id:
                break

            x_min, y_min, x_max, y_max = np.array(e[1:5], np.int32)
            score = e[5]
            label = int(e[6])
            if score > threshold[label]:
                x_c = (x_max + x_min) / 2
                y_c = (y_max + y_min) / 2
                w = x_max - x_min
                h = y_max - y_min
                bboxes.append((x_min, y_min, x_max, y_max))
                confidents.append(score)
                labels.append(label)
                dets.append([x_min, y_min, x_max, y_max, label])

            event_id += 1
            if event_id >= len(events):
                break
    except:
        print('error')

    bboxes = np.array(bboxes)
    confidents = np.array(confidents)
    labels = np.array(labels)
    dets = np.array(dets)

    if len(bboxes) > 0:
        bboxes, labels, confidents = hard_nms(bboxes, labels, confidents, 0.3)
        # from tlbr to tlwh
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]
    
    features = encoder(image, bboxes)
    detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, confidents, features)]

    tracker.predict()
    tracker.update(detections)
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        x_min, y_min, x_max, y_max = track.to_tlbr()
        track_id = track.track_id

        label = find_tracking_label([x_min, y_min, x_max, y_max], dets)

        if track_id not in track_dict.keys():
            track_dict[track_id] = [(x_min, y_min, x_max, y_max, frame_id, label)]
            track_colors[track_id] = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            track_labels[track_id] = {
                -1 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0,
            }
        else:
            x1, y1, x2, y2, _, _ = track_dict[track_id][-1]
            x_c1 = (x1 + x2) / 2
            y_c1 = (y1 + y2) / 2
            track_dict[track_id].append((x_min, y_min, x_max, y_max, frame_id, label))
            x_c2 = (x_min + x_max) / 2
            y_c2 = (y_min + y_max) / 2
            x_c1, y_c1, x_c2, y_c2 = np.array([x_c1, y_c1, x_c2, y_c2], np.int32)
            frame = cv2.line(frame, (x_c1, y_c1), (x_c2, y_c2), track_colors[track_id], 2)
        
        track_labels[track_id][label] += 1
        for i in range(1,5):
            if track_labels[track_id][label] < track_labels[track_id][i]:
                label = i

        x_min, y_min, x_max, y_max = np.array([x_min, y_min, x_max, y_max], np.int32)
        # frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), track_colors[track_id], 2)
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), label_colors[label], 2)

    cv2.imshow('', frame)
    if cv2.waitKey(1) == 27:
        break
    frame_id += 1
    if event_id >= len(events):
        break
