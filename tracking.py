import cv2
import numpy as np
import json
from sort import *
import random

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

label_colors = {
    1 : (0,0,0),
    2 : (255,0,0),
    3 : (0,255,0),
    4 : (0,0,255),
}

threshold = {
    1 : 0.2,
    2 : 0.3,
    3 : 0.3,
    4 : 0.3
}

cam_num = 'cam_06'

f = open('datasets/{}.txt'.format(cam_num))
events = f.readlines()

f = json.load(open('datasets/{}.json'.format(cam_num)))
zone = f['shapes'][0]['points']
zone = np.array(zone, np.int32)

event_id = 0

# Definition of the parameters
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0

#initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

tracker = Tracker(metric)
# tracker = [None]
# for i in range(1,5):
#     tracker.append(Tracker(metric))

frame_id = 0

track_dict = {}
track_colors = {}

cap = cv2.VideoCapture('datasets/{}.mp4'.format(cam_num))
while cap.isOpened():
    ret, frame = cap.read()

    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = np.array(image, np.float32) / 255.

    frame = cv2.polylines(frame, [zone], True, (127,127,127), 5)

    dets = []
    bboxes = []
    confidents = []
    labels = []

    try:
        while True:
            e = list(map(float, events[event_id].split(' ')))
            if e[0] != frame_id:
                break

            x_min, y_min, x_max, y_max = np.array(e[1:5], np.int32)
            score = e[5]
            label = int(e[6])
            if score > threshold[label]:
                # frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), label_colors[label], 2)
                dets.append((x_min, y_min, x_max, y_max, score))
                x_c = (x_max + x_min) / 2
                y_c = (y_max + y_min) / 2
                w = x_max - x_min
                h = y_max - y_min
                bboxes.append((x_min, y_min, w, h))
                confidents.append(score)

            event_id += 1
            if event_id >= len(events):
                break
    except:
        print('error')

    bboxes = np.array(bboxes)
    features = encoder(image, bboxes)
    confidents = np.array(confidents)
    dets = np.array(dets)
    detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, confidents, features)]

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        x_min, y_min, x_max, y_max = track.to_tlbr()
        track_id = track.track_id

        if track_id not in track_dict.keys():
            track_dict[track_id] = [(x_min, y_min, x_max, y_max, frame_id)]
            track_colors[track_id] = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        else:
            x1, y1, x2, y2, _ = track_dict[track_id][-1]
            x_c1 = (x1 + x2) / 2
            y_c1 = (y1 + y2) / 2
            track_dict[track_id].append((x_min, y_min, x_max, y_max, frame_id))
            x_c2 = (x_min + x_max) / 2
            y_c2 = (y_min + y_max) / 2
            x_c1, y_c1, x_c2, y_c2 = np.array([x_c1, y_c1, x_c2, y_c2], np.int32)
            frame = cv2.line(frame, (x_c1, y_c1), (x_c2, y_c2), track_colors[track_id], 2)

        x_min, y_min, x_max, y_max = np.array([x_min, y_min, x_max, y_max], np.int32)
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), track_colors[track_id], 2)

    cv2.imshow('', frame)
    if cv2.waitKey(1) == 27:
        break
    frame_id += 1
    if event_id >= len(events):
        break
