from collections import namedtuple
from typing import List

import numpy as np
from sahi.prediction import PredictionResult
from trackers.byte_tracker.byte_tracker import BYTETracker
from trackers.ocsort_tracker.ocsort import OCSort

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class ByteTrack:
    def __init__(self, accumulate_results, args):
        # Convert args to nametuple as the ByteTrack implementation accesses the args as an attribute
        Args = namedtuple('ByteTrackArgs', ['frame_rate', 'track_thresh', 'track_buffer', 'match_thresh', 'mot20'])
        args=Args(args['frame_rate'],args['track_thresh'],args['track_buffer'],args['match_thresh'], False)

        self.tracker = BYTETracker(args, args.frame_rate)

        self.accumulate_results = accumulate_results
        self.matrix_predictions = np.empty((0, 10), dtype=float)
        self.frame_number = 0

        self.name = self.create_tracker_name()

    def offline_tracking(self, object_prediction_list: PredictionResult):
        raise NotImplementedError("The offline tracking is not implemented.")

    def update_online(self, object_prediction_list: List[PredictionResult]):
        self.frame_number += 1
        if len(object_prediction_list) > 0:
            dets = np.array([prediction.bbox.to_xyxy() for prediction in object_prediction_list])
            categories = np.array([prediction.category.id for prediction in object_prediction_list])
            scores = np.array([[prediction.score.value for prediction in object_prediction_list]]).reshape(-1, 1)
            dets = np.hstack((dets, scores))
        else:
            dets = np.empty(((0, 5)), dtype=float)
            categories = np.empty(((0, 1)), dtype=int)
            scores = np.empty(((0, 1)), dtype=float)

        tracked_objects = self.tracker.update(dets, (3840, 2160), (3840, 2160))

        tracked_objects = np.array(tracked_objects)

        #if self.accumulate_results and tracked_objects.shape[0] > 0:
        #    self.accumulate(tracked_objects)

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

    def create_tracker_name(self):
        name = f"ByteTrack"
        return name