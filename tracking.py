import cv2
import numpy as np
import json
import random

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from utils import *

label_colors = {
    -1: (127,127,127),
    1 : (0,0,0),
    2 : (255,0,0),
    3 : (0,255,0),
    4 : (0,0,255),
}

threshold = {
    1 : 0.1,
    2 : 0.2,
    3 : 0.3,
    4 : 0.3
}

cam_num = 'cam_03'

f = open('datasets/d4/{}.txt'.format(cam_num))
events = f.readlines()

polygon, paths = load_zone_anno('datasets/{}.json'.format(cam_num))
polygon = np.array(polygon)

with open('results/result_{}.txt'.format(cam_num)) as f:
    result_lines = f.readlines()

counting_event = {}

for r in result_lines:
    cam_num = r.split(' ')[0]
    frame_id = int(r.split(' ')[1])
    movement_id = int(r.split(' ')[2])
    vehicle_class_id = int(r.split(' ')[3])
    if frame_id not in counting_event.keys():
        counting_event[frame_id] = {}
        for i in range(1, len(paths) + 1):
            counting_event[frame_id][i] = {
                1:0, 2:0, 3:0, 4:0
            }
    counting_event[frame_id][movement_id][vehicle_class_id] += 1

counting = {}
for i in range(1, len(paths) + 1):
    counting[i] = {
        1:0, 2:0, 3:0, 4:0
    }

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

    frame = cv2.polylines(frame, [polygon], True, (127,127,127), 5)

    if frame_id + 1 in counting_event.keys():
        for m_id in range(1, len(paths) + 1):
            for i in range(1, 5):
                counting[m_id][i] += counting_event[frame_id + 1][m_id][i]

    # for c in counting:
    #     print(c, counting[c])
    # print('----------------')

    for movement_id, path in paths.items():
        frame = cv2.line(frame, (path[0][0], path[0][1]), (path[1][0], path[1][1]), path[2], 5)
        movement_id = int(movement_id)
        a, b, c, d = counting[movement_id].values()
        frame = cv2.putText(frame, 
                            '  {} {} {} {}'.format(a, b, c, d), 
                            (path[1][0], path[1][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, path[2], 2)

    bboxes = []
    confidents = []
    labels = []
    dets = []

    while True:
        e = list(map(float, events[event_id].split(' ')))
        if e[0] != frame_id:
            break

        x_min, y_min, x_max, y_max = np.array(e[1:5], np.int32)

        if check_bbox_intersect_polygon(polygon, (x_min, y_min, x_max, y_max)):
            score = e[5]
            label = int(e[6])
            if score > threshold[label]:
                bboxes.append((x_min, y_min, x_max, y_max))
                confidents.append(score)
                labels.append(label)
                dets.append([x_min, y_min, x_max, y_max, label])

        event_id += 1
        if event_id >= len(events):
            break

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
            track_dict[track_id].append((x_min, y_min, x_max, y_max, frame_id, label))
            l_track = len(track_dict[track_id])
            for i in range(max(1,l_track - 10), l_track):
                x_min, y_min, x_max, y_max, _, _ = track_dict[track_id][i - 1]
                x_c1 = int((x_min + x_max) / 2)
                y_c1 = int((y_min + y_max) / 2)
                x_min, y_min, x_max, y_max, _, _ = track_dict[track_id][i]
                x_c2 = int((x_min + x_max) / 2)
                y_c2 = int((y_min + y_max) / 2)
                frame = cv2.line(frame, (x_c1, y_c1), (x_c2, y_c2), track_colors[track_id], 2)
        
        track_labels[track_id][label] += 1
        for i in range(1,5):
            if track_labels[track_id][label] < track_labels[track_id][i]:
                label = i

        x_min, y_min, x_max, y_max = np.array([x_min, y_min, x_max, y_max], np.int32)
        # frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), track_colors[track_id], 2)
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), label_colors[label], 2)

    cv2.imshow(cam_num, frame)
    if cv2.waitKey(1) == 27:
        break
    frame_id += 1
    if event_id >= len(events):
        break
