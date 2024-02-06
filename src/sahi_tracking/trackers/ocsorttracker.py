from typing import List

import numpy as np
from sahi.prediction import PredictionResult
from trackers.ocsort_tracker.ocsort import OCSort


class OcSortTracker:
    def __init__(self, accumulate_results, det_thresh, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        self.tracker = OCSort(det_thresh, max_age, min_hits, iou_threshold, delta_t, asso_func, inertia, use_byte)

        self.accumulate_results = accumulate_results
        self.matrix_predictions = np.empty((0, 10), dtype=float)
        self.frame_number = 0

        self.name = self.create_tracker_name(asso_func, use_byte)

    def offline_tracking(self, object_prediction_list: PredictionResult):
        raise NotImplementedError("The offline tracking is not implemented.")

    def update_online(self, object_prediction_list: List[PredictionResult]):
        self.frame_number += 1
        if len(object_prediction_list) > 0:
            dets = np.array([prediction.bbox.to_xyxy() for prediction in object_prediction_list])
            scores = np.array([[prediction.score.value for prediction in object_prediction_list]]).reshape(-1, 1)
            dets = np.hstack((dets, scores))
        else:
            dets = np.empty(((0, 5)), dtype=float)

        tracked_objects = self.tracker.update(dets, (3840, 2160), (3840, 2160))

        if self.accumulate_results and tracked_objects.shape[0] > 0:
            self.accumulate(tracked_objects)

        return tracked_objects

    def accumulate(self, tracked_objects):
        trk_num = tracked_objects.shape[0]
        boxes = tracked_objects[:, :4]
        ids = tracked_objects[:, 4]

        for trk in range(trk_num):
            new_row = [
                int(self.frame_number),
                int(ids[trk]),
                boxes[trk][0],
                boxes[trk][1],
                boxes[trk][2],
                boxes[trk][3],
                1,
                -1,
                -1,
                -1,
            ]
            self.matrix_predictions = np.vstack((self.matrix_predictions, new_row))

    def get_mot_list(self):
        return self.matrix_predictions

    def create_tracker_name(self, asso_func, use_byte):
        name = f"OC_SORT ({asso_func})"
        if use_byte == True:
            name = name + " + BYTE"
        return name
