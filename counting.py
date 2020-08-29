import os

import numpy as np
import cv2

import json

from utils import *

cam_num = 'cam_02'

label_colors = {
    -1: (127,127,127),
    1 : (0,0,0),
    2 : (255,0,0),
    3 : (0,255,0),
    4 : (0,0,255),
}

cosin_thresh = 0.7
h_dist_thresh = 4000

def counting_moi(paths, vector_list):
    """
    Args:
        paths: List of MOI - (first_point, last_point)
        moto_vector_list: List of tuples (first_point, last_point, last_frame_id) 
    
    Returns:
        A list of tuples (frame_id, movement_id, vehicle_class_id)
    """
    moi_detection_list = []
    for vector in vector_list:
        vehicle_id = vector[3]
        max_cosin = -2
        movement_id = ''
        last_frame = 0
        for movement_label, movement_vector in paths.items():
            cosin = cosin_similarity(movement_vector, vector)
            if cosin > max_cosin:
                max_cosin = cosin
                movement_id = movement_label
                last_frame = vector[2]
                h_dist = hausdorff_distance(vector, movement_vector)
                e_dist = euclidean_distance(vector[0], vector[1])
                movement_length = euclidean_distance(movement_vector[0], movement_vector[1])
        # mind this part
        if max_cosin < cosin_thresh or h_dist > h_dist_thresh or e_dist < movement_length * 0.2:
            continue
        moi_detection_list.append((last_frame, movement_id, vehicle_id))
    return moi_detection_list

track_dict = {}
track_colors = {}
track_label_counter = {}
track_label = {}

frame_track = {}

with open('tracking_results_v3/tracking_{}.txt'.format(cam_num)) as f:
    tracks = f.readlines()
    for t in tracks:
        track_id, x_min, y_min, x_max, y_max, frame_id, label = list(map(int, map(float, t.split(' '))))
        if track_id not in track_dict.keys():
            track_dict[track_id] = {
                'bbox' : [[x_min, y_min, x_max, y_max]],
                'frame_id' : [frame_id],
                'label' : [label],
                'bbox_by_frame_id' : {}
            }
            track_colors[track_id] = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            track_label_counter[track_id] = {
                -1 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0
            }
        else:
            if frame_id - track_dict[track_id]['frame_id'][-1] < 1000:
                track_dict[track_id]['bbox'].append([x_min, y_min, x_max, y_max])
                track_dict[track_id]['frame_id'].append(frame_id)
                track_dict[track_id]['label'].append(label)
            else:
                track_dict[track_id] = {
                    'bbox' : [[x_min, y_min, x_max, y_max]],
                    'frame_id' : [frame_id],
                    'label' : [label],
                    'bbox_by_frame_id' : {}
                }
                track_label_counter[track_id] = {
                    -1 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0
                }

        track_dict[track_id]['bbox_by_frame_id'][frame_id] = [x_min, y_min, x_max, y_max]

        track_label_counter[track_id][label] += 1

        for i in range(1,5):
            if track_label_counter[track_id][label] < track_label_counter[track_id][i]:
                label = i

        track_label[track_id] = label

        if frame_id not in frame_track.keys():
            frame_track[frame_id] = []
        frame_track[frame_id].append(track_id)

polygon, paths = load_zone_anno('datasets/{}.json'.format(cam_num))
polygon = np.array(polygon)

vector_list = []
for tracker_id, tracker_list in track_dict.items():
    if len(tracker_list) > 1:
        first = tracker_list['bbox'][0]
        last = tracker_list['bbox'][-1]
        first_point = ((first[2] + first[0])/2, (first[3] + first[1])/2)
        last_point = ((last[2] + last[0])/2, (last[3] + last[1])/2)
        first_frame = track_dict[tracker_id]['frame_id'][0]
        last_frame = track_dict[tracker_id]['frame_id'][-1]
        if last_frame - first_frame < 5:
            continue
        vector_list.append((first_point, last_point, last_frame, track_label[tracker_id]))

moi_detections = counting_moi(paths, vector_list)

print(len(moi_detections))

counting_event = {}

for m in moi_detections:
    frame_id = m[0]
    movement_id = int(m[1])
    vehicle_id = int(m[2])
    if frame_id not in counting_event.keys():
        counting_event[frame_id] = {}
        for i in range(1, len(paths) + 1):
            counting_event[frame_id][i] = {
                -1:0, 1:0, 2:0, 3:0, 4:0
            }
    counting_event[frame_id][movement_id][vehicle_id] += 1

cap = cv2.VideoCapture('datasets/{}.mp4'.format(cam_num))

frame_id = 0

counting = {}
for i in range(1, len(paths) + 1):
    counting[i] = {
        1:0, 2:0, 3:0, 4:0
    }

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.polylines(frame, [polygon], True, (127,127,127), 5)

    if frame_id + 1 in counting_event.keys():
        for m_id in range(1, len(paths) + 1):
            for i in range(1, 5):
                counting[m_id][i] += counting_event[frame_id + 1][m_id][i]

    for movement_id, path in paths.items():
        frame = cv2.line(frame, (path[0][0], path[0][1]), (path[1][0], path[1][1]), path[2], 5)
        frame = cv2.circle(frame, (path[0][0], path[0][1]), h_dist_thresh, (0,0,0), 5)
        frame = cv2.circle(frame, (path[1][0], path[1][1]), h_dist_thresh, (0,0,0), 5)
        movement_id = int(movement_id)
        a, b, c, d = counting[movement_id].values()
        frame = cv2.putText(frame, 
                            '  {} {} {} {}'.format(a, b, c, d), 
                            (path[1][0], path[1][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, path[2], 2)

    if frame_id in frame_track.keys():
        for track_id in frame_track[frame_id]:
            try:
                x_min, y_min, x_max, y_max = track_dict[track_id]['bbox_by_frame_id'][frame_id]
                label = track_label[track_id]
                frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), label_colors[label], 2)

                l_track = len(track_dict[track_id]['bbox'])
                for i in range(1, l_track):
                    x_min, y_min, x_max, y_max = track_dict[track_id]['bbox'][i - 1]
                    x_c1 = int((x_min + x_max) / 2)
                    y_c1 = int((y_min + y_max) / 2)
                    x_min, y_min, x_max, y_max = track_dict[track_id]['bbox'][i]
                    x_c2 = int((x_min + x_max) / 2)
                    y_c2 = int((y_min + y_max) / 2)
                    frame = cv2.line(frame, (x_c1, y_c1), (x_c2, y_c2), track_colors[track_id], 2)
            except:
                pass

    cv2.imshow(cam_num, frame)
    if cv2.waitKey(0) == 27:
        break

    frame_id += 1