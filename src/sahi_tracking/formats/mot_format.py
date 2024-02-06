from pathlib import Path
from typing import List, Tuple

import numpy as np
from sahi.prediction import PredictionResult, ObjectPrediction


def create_mot_folder_structure(name: str, path: Path):
    """
    reates a folder structure for the MOT dataset.
    """
    mot_path = Path(path) / f"{name}"
    mot_path.mkdir(parents=True, exist_ok=True)
    mot_img_path = mot_path / "img1"
    mot_img_path.mkdir(parents=True, exist_ok=True)
    det_path = Path(f"{mot_path}/det/det.txt")
    gt_path = Path(f"{mot_path}/gt/gt.txt")
    det_path.parents[0].mkdir(parents=True, exist_ok=True)
    gt_path.parents[0].mkdir(parents=True, exist_ok=True)

    return mot_path, mot_img_path, det_path, gt_path


def mot_matrix_from_sahi_object_prediction_list(sequence_name: str, frame_pred: List[Tuple[str, ObjectPrediction]],
                                                output_path: Path):
    mot_matrix_preds = []
    for frame_id, object_prediction_list in frame_pred:

        for object_prediction in object_prediction_list:
            bbox = object_prediction.bbox
            detection_as_xyxy = bbox.to_voc_bbox()

            new_row = [
                int(frame_id),
                -1,
                detection_as_xyxy[0],
                detection_as_xyxy[1],
                detection_as_xyxy[2],
                detection_as_xyxy[3],
                float(object_prediction.score.value)
            ]
            if np.shape(mot_matrix_preds)[0] == 0:
                mot_matrix_preds = new_row
            else:
                mot_matrix_preds = np.vstack((mot_matrix_preds, new_row))

    if output_path:
        output_path.mkdir(exist_ok=True, parents=True)
        np.savetxt(output_path / f"{sequence_name}.txt", mot_matrix_preds, delimiter=",")

    return mot_matrix_preds
