# -*- coding: utf-8 -*-
from yolo.backend.utils.eval._box_match import BoxMatcher

def count_true_positives(detect_boxes, true_boxes, detect_labels=None, true_labels=None):
    """
    # Args
        detect_boxes : array, shape of (n_detected_boxes, 4)
        true_boxes : array, shape of (n_true_boxes, 4)
        detected_labels : array, shape of (n_detected_boxes,)
        true_labels :
    """
    n_true_positives = 0
 	print("Detect boxes:",detect_boxes,"True boxes", true_boxes)
    matcher = BoxMatcher(detect_boxes, true_boxes, detect_labels, true_labels)
    for i in range(len(detect_boxes)):
        matching_idx, iou = matcher.match_idx_of_box1_idx(i)
        # print("detect_idx: {}, true_idx: {}, matching-score: {}".format(i, matching_idx, iou))
        if matching_idx is not None and iou > 0.5:
            n_true_positives += 1
    return n_true_positives


def calc_score(n_true_positives, n_truth, n_pred):
    """
    # Args
        detect_boxes : list of box-arrays
        true_boxes : list of box-arrays
    """
    print("True positives:", n_true_positives)
    print("n_pred:", n_pred)
    print("n_truth:", n_truth)
    precision = n_true_positives / n_pred
    recall = n_true_positives / n_truth
    print("Precision:", precision)
    print("Recall:", recall)
    fscore = 2* precision * recall / (precision + recall)
    score = {"fscore": fscore, "precision": precision, "recall": recall}
    return score
    

if __name__ == '__main__':
    pass
