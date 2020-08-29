import numpy as np
import random

import json

import bb_polygon

def check_bbox_intersect_polygon(polygon, bbox):
  """
  
  Args:
    polygon: List of points (x,y)
    bbox: A tuple (xmin, ymin, xmax, ymax)
  
  Returns:
    True if the bbox intersect the polygon
  """
  x1, y1, x2, y2 = bbox
  bb = [(x1,y1), (x2, y1), (x2,y2), (x1,y2)]
  return bb_polygon.is_bounding_box_intersect(bb, polygon)

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