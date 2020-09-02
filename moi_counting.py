import numpy as np
import cv2
import json
import random

from utils import *

def read_track_file(track_file_name):
    track_dict = {}
    track_label = {}
    track_label_counter = {}
    
    with open(track_file_name) as track_file:
        tracks = track_file.readlines()
        for tr in tracks:
            track_id = int(tr.split(' ')[0])
            x_min, y_min, x_max, y_max = list(map(int, map(float, tr.split(' ')[1:5])))
            frame_id = int(tr.split(' ')[5])
            label = int(tr.split(' ')[6])

            if track_id not in track_dict.keys():
                track_dict[track_id] = {
                    'bbox': [],
                    'frame_id': [],
                    'label': label,
                }
                track_label_counter[track_id] = {
                    -1 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0
                }
            track_dict[track_id]['bbox'].append([x_min, y_min, x_max, y_max])
            track_dict[track_id]['frame_id'].append(frame_id)
            track_label_counter[track_id][label] += 1
            for i in range(1, 5):
                if track_label_counter[track_id][label] < track_label_counter[track_id][i]:
                    label = i
            track_label[track_id] = label

    return track_dict, track_label


def get_vector_list(polygon, mask, track_dict, track_label):
    vector_list = []

    for track_id, track_list in track_dict.items():
        len_track = len(track_list['bbox'])

        track_points = np.array(track_list['bbox'])
        center_points = np.zeros((track_points.shape[0],2))
        center_points[:,0] = (track_points[:,0] + track_points[:,2]) / 2
        center_points[:,1] = (track_points[:,1] + track_points[:,3]) / 2

        if not check_track_intersect_roi(center_points, mask):
            continue

        flag = True
        
        for i in range(len_track):
            first_bbox = track_list['bbox'][i]
            if check_bbox_intersect_polygon(first_bbox, polygon, mask):
                flag = False
                first_point = ((first_bbox[0] + first_bbox[2]) / 2, (first_bbox[1] + first_bbox[3]) / 2)
                first_frame = track_list['frame_id'][i]
                for j in range(i, len_track):
                    last_bbox = track_list['bbox'][j]
                    last_point = ((last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2)
                    last_frame = track_list['frame_id'][j]
                    if not check_bbox_intersect_polygon(last_bbox, polygon, mask):
                        break
                break
        if flag or last_frame - first_frame < 5:
            continue
        if check_point_inside_roi(first_point, mask) and \
            check_point_inside_roi(last_point, mask):
            d0 = 10000000
            d1 = 10000000
            for i in range(len(polygon)):
                d0 = np.minimum(d0, distance_point_to_line(first_point, [polygon[i-1], polygon[i]]))
                d1 = np.minimum(d1, distance_point_to_line(last_point, [polygon[i-1], polygon[i]]))
            if d0 > 20 and d1 > 20:
                continue
        if check_point_inside_roi(first_point, mask) and \
            check_point_inside_roi(last_point, mask):
            continue
        vector_list.append((first_point, last_point, last_frame, track_label[track_id]))

    return vector_list

def filter_vector(vector_list, paths, mask):
    filtered_list = []
    for v in vector_list:
        p0, p1, last_frame, label = v
        p0 = np.array(p0, np.int32)
        p1 = np.array(p1, np.int32)
        for movement_id, movement_vector in paths.items():
            cosin = cosin_similarity([p0, p1], movement_vector[0:2])
            d0 = distance_point_to_line(p0, movement_vector[0:2])
            d1 = distance_point_to_line(p1, movement_vector[0:2])
            d = np.maximum(d0, d1)
            norm = np.sqrt((p0 - p1).dot(p0 - p1))
            if cosin >= 0.7 and d < 350 and norm > 50:
                filtered_list.append(v)
    return filtered_list

def counting_moi(cam_num, polygon, paths, track_dict, track_label):
    cosin_thresh = 0.7
    h_dist_thresh = 4000

    mask = np.zeros((720, 1280), np.uint8)
    mask = cv2.fillPoly(mask, [polygon], 255)

    vector_list = get_vector_list(polygon, mask, track_dict, track_label)

    vector_list = filter_vector(vector_list, paths, mask)
    moi_detection_list = []

    # image = np.ones((720, 1280, 3), np.uint8) * 255
    # image = cv2.fillPoly(image, [polygon], (127,127,127))

    for vector in vector_list:
        last_frame = vector[2]
        vehicle_id = vector[3]
        flag = True
        for movement_label, movement_vector in paths.items():
            if check_line_intersect_segment((vector[0], vector[1]), movement_vector[3]) and \
                check_line_intersect_segment((vector[0], vector[1]), movement_vector[4]):
                cosin = cosin_similarity(movement_vector, vector)
                if cosin < cosin_thresh:
                    continue
                flag = False
                movement_id = movement_label
                last_frame = vector[2]
                h_dist = hausdorff_distance(vector, movement_vector)
                e_dist = euclidean_distance(vector[0], vector[1])
                movement_length = euclidean_distance(movement_vector[0], movement_vector[1])
        if flag:
            if cam_num in ['cam_01', 'cam_02', 'cam_03']:
                movement_label = '1'
                movement_vector = paths['1']
                start_line = [polygon[1], polygon[2]]
                end_line = [polygon[-1], polygon[0]]
                if check_line_intersect_segment((vector[0], vector[1]), start_line) and \
                    check_line_intersect_segment((vector[0], vector[1]), end_line):
                    cosin = cosin_similarity(movement_vector, vector)
                    if cosin >= cosin_thresh:
                        flag = False
                        movement_id = movement_label
                        last_frame = vector[2]
                        h_dist = hausdorff_distance(vector, movement_vector)
                        e_dist = euclidean_distance(vector[0], vector[1])
                        movement_length = euclidean_distance(movement_vector[0], movement_vector[1])
            if cam_num in ['cam_10']:
                movement_label = '2'
                movement_vector = paths['2']
                start_line = [polygon[3], polygon[4]]
                end_line = [polygon[1], polygon[2]]
                if check_line_intersect_segment((vector[0], vector[1]), start_line) and \
                    check_line_intersect_segment((vector[0], vector[1]), end_line):
                    cosin = cosin_similarity(movement_vector, vector)
                    if cosin >= cosin_thresh:
                        flag = False
                        movement_id = movement_label
                        last_frame = vector[2]
                        h_dist = hausdorff_distance(vector, movement_vector)
                        e_dist = euclidean_distance(vector[0], vector[1])
                        movement_length = euclidean_distance(movement_vector[0], movement_vector[1])

        if flag or h_dist > h_dist_thresh or e_dist < movement_length * 0.1:
            continue
        # if movement_id == '3':
        #     color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        #     image = cv2.line(image, (int(vector[0][0]), int(vector[0][1])), 
        #         (int(vector[1][0]), int(vector[1][1])), color, 2)
        moi_detection_list.append((last_frame, movement_id, vehicle_id))
    # cv2.imshow('', image)
    # cv2.waitKey(0)
    return moi_detection_list


def main():
    for cam_id in range(1,26):
        cam_num = 'cam_{:02d}'.format(cam_id)
        track_dict, track_label = read_track_file('tracking_results/v2.0/tracking_{}.txt'.format(cam_num))
        polygon, paths = load_zone_anno('datasets/{}.json'.format(cam_num))
        polygon = np.array(polygon, np.int32)
        moi_detections = counting_moi(cam_num, polygon, paths, track_dict, track_label)
        with open('results/v2.0/result_{}.txt'.format(cam_num), 'w') as result_file:
            for frame_id, movement_id, vehicle_id in moi_detections:
                if vehicle_id == -1:
                    continue
                result_file.write('{} {} {} {}\n'.format(cam_num, frame_id + 1, movement_id, vehicle_id))
        print(cam_num, len(moi_detections))

if __name__ == "__main__":
    main()