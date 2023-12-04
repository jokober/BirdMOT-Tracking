from typing import List
import numpy as np
from norfair import Detection, Tracker
from norfair.filter import OptimizedKalmanFilterFactory
from sahi.prediction import PredictionResult


class NorfairTracker:
    def __init__(self, accumulate_results, initialization_delay, distance_function, hit_counter_max, distance_threshold, **kwargs):
        self.tracker = Tracker(
            initialization_delay=initialization_delay,
            distance_function=distance_function,
            hit_counter_max=hit_counter_max,
            filter_factory=OptimizedKalmanFilterFactory(),
            distance_threshold=distance_threshold,
        )

        self.accumulate_results = accumulate_results
        self.matrix_predictions = np.empty((0,10), dtype=float)
        self.frame_number = 0

        self.name = self.create_tracker_name(distance_function)

    def offline_tracking(self, object_prediction_list: PredictionResult):
        raise NotImplementedError("The offline tracking is not implemented.")


    def update_online(self, object_prediction_list: PredictionResult):
        self.frame_number += 1
        detections = self.get_detections(object_prediction_list)
        tracked_objects = self.tracker.update(detections=detections)

        if self.accumulate_results:
            self.accumulate(tracked_objects)

        return tracked_objects

    def get_detections(self, object_prediction_list: List[PredictionResult]) -> List[Detection]:
        detections = []
        for prediction in object_prediction_list:
            bbox = prediction.bbox

            detection_as_xyxy = bbox.to_voc_bbox()
            bbox = np.array(
                [
                    [detection_as_xyxy[0], detection_as_xyxy[1]],
                    [detection_as_xyxy[2], detection_as_xyxy[3]],
                ]
            )
            detections.append(
                Detection(
                    points=bbox,
                    scores=np.array([prediction.score.value for _ in bbox]),
                    label=prediction.category.id,
                )
            )
        return detections

    def accumulate(self, tracked_objects):
        for obj in tracked_objects:
            new_row = [
                self.frame_number,
                obj.id,
                obj.estimate[0, 0],
                obj.estimate[0, 1],
                obj.estimate[1, 0],
                obj.estimate[1, 1],
                -1,
                -1,
                -1,
                -1,
            ]
            self.matrix_predictions = np.vstack((self.matrix_predictions, new_row))

    def get_mot_list(self):
        return self.matrix_predictions

    def create_tracker_name(self, distance_function):
        return f"Norfair ({distance_function})"