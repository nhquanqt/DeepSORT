import numpy as np
import random

import json

import bb_polygon

def check_bbox_intersect_polygon(bbox, polygon, mask):
    polygon_t = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
    ]
    for p in polygon_t:
        if check_point_inside_roi(p, mask):
            return True
    flag = True
    for pt in polygon:
        if pt[1] > bbox[1]:
            flag = False
    if flag:
        return False
    flag = True
    for pt in polygon:
        if pt[1] < bbox[3]:
            flag = False
    if flag:
        return False
    flag = True
    for pt in polygon:
        if pt[0] > bbox[0]:
            flag = False
    if flag:
        return False
    flag = True
    for pt in polygon:
        if pt[0] < bbox[2]:
            flag = False
    if flag:
        return False
    return check_two_polygon_intersect(polygon, polygon_t)

def check_two_polygon_intersect(polygon_1, polygon_2):
    for i in range(len(polygon_1)):
        line = (polygon_1[i-1], polygon_1[i])
        flag = True
        for p1 in polygon_1:
            for p2 in polygon_2:
                if not check_line_intersect_segment(line, [p1, p2]):
                    flag = False
        if flag:
            return False
    return True


def check_track_intersect_roi(track_list, mask):
    for p in track_list:
        if check_point_inside_roi(p, mask):
            return True
    return False

def check_bbox_outside_roi(bbox, mask):
    points = [  
        [bbox[0], bbox[1]],
        [bbox[0], bbox[3]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    ]
    for p in points:
        if check_point_inside_roi(p, mask):
            return False
    return True

def distance_point_to_line(point, line):
    v = np.array([line[1][0] - line[0][0], line[1][1] - line[0][1]])
    c =  - line[1][0] * v[1] + line[1][1] * v[0]
    return abs(point[0] * v[1] - point[1] * v[0] + c) / np.sqrt(v.dot(v))

def check_line_intersect_segment(line, segment):
    '''
    Both line and segment presented by two points (x,y)
    '''
    v = np.array([line[1][0] - line[0][0], line[1][1] - line[0][1]])
    c =  - line[1][0] * v[1] + line[1][1] * v[0]
    t0 = segment[0][0] * v[1] - segment[0][1] * v[0] + c
    t1 = segment[1][0] * v[1] - segment[1][1] * v[0] + c
    return (t0 * t1 <= 0)


def intersect_area(bbox, polygon):
    bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    # TODO:

def check_point_inside_roi(point, mask):
    point = np.array(point, np.int32)
    if point[0] < 0 or point[0] >= mask.shape[1]:
        return False
    if point[1] < 0 or point[1] >= mask.shape[0]:
        return False
    return mask[point[1],point[0]] != 0

def euclidean_distance(p1, p2):
    vec = np.array(p1) - np.array(p2)
    return np.sqrt(vec.dot(vec))

def hausdorff_distance(a2d, b2d):
    d_max = 0
    for i in range(2):
        d_min = np.minimum(euclidean_distance(a2d[i],b2d[0]), 
                            euclidean_distance(a2d[i], b2d[1]))
        d_max = np.maximum(d_max, d_min)
    return d_max


def cosin_similarity(a2d, b2d):  
    a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1] - a2d[0][1]))
    b = np.array((b2d[1][0] - b2d[0][0], b2d[1][1] - b2d[0][1]))
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

# def check_bbox_intersect_polygon(bbox, polygon):
#   """
  
#   Args:
#     polygon: List of points (x,y)
#     bbox: A tuple (xmin, ymin, xmax, ymax)
  
#   Returns:
#     True if the bbox intersect the polygon
#   """
#   x1, y1, x2, y2 = bbox
#   bb = [(x1,y1), (x2, y1), (x2,y2), (x1,y2)]
#   return bb_polygon.is_bounding_box_intersect(bb, polygon)

def load_zone_anno(json_filename):
    """
    Load the json with ROI and MOI annotation.

    """
    with open(json_filename) as jsonfile:
        dd = json.load(jsonfile)
        polygon = [(int(x), int(y)) for x, y in dd['shapes'][0]['points']]
        paths = {}
        for it in dd['shapes'][1:]:
            kk = str(int(it['label'][-2:]))
            paths[kk] = [(int(x), int(y)) for x, y
                    in it['points']]
            paths[kk].append((random.randint(0,256), random.randint(0,256), random.randint(0,256)))
            for i in range(len(polygon)):
                p0 = polygon[i-1]
                p1 = polygon[i]
                if check_line_intersect_segment(paths[kk][0:2], (p0,p1)):
                    paths[kk].append([p0,p1])
            d0 = distance_point_to_line(paths[kk][0], paths[kk][-1])
            d1 = distance_point_to_line(paths[kk][1], paths[kk][-1])
            if d0 < d1:
                tmp = paths[kk][-1]
                paths[kk][-1] = paths[kk][-2]
                paths[kk][-2] = tmp
    return polygon, paths

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