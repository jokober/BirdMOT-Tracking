import time
from pathlib import Path

import cv2
import numpy as np
from sahi.prediction import ObjectPrediction

from sahi_tracking.background_substraction.backsub import get_detections_backsub


def backsub_on_mot_data(sequence_name: str, mot_path: Path, output_path: Path, return_sahi_obj_pred_list, algo,
                        var_thresh, detectShadows, shadow_thresh, history, dist2Threshold, bbox_thresh, nms_thresh,
                        kernel, **vars):
    kernel = np.array((kernel, kernel), dtype=np.uint8)
    mot_matrix_preds = []
    sahi_obj_pred_list = []
    compute_time_agg = 0

    if algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=var_thresh, detectShadows=detectShadows)
        backSub.setShadowThreshold(shadow_thresh)
        backSub.setHistory(history)

    elif algo == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=dist2Threshold, detectShadows=True)

    img_list = sorted(list((mot_path / "img1").glob('*.png')) + list((mot_path / "img1").glob('*.jpg')))
    for frame_id, img in enumerate(img_list, 1):
        frame_bgr = cv2.imread(img.as_posix())
        # MOT - 1654911941_zoex_0446964_0447233
        # MOT - 1654911941_zoex_0447234_0447511

        # time when we finish processing for this frame
        new_frame_time = time.time()

        # get detections
        detections = get_detections_backsub(backSub,
                                            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                                            bbox_thresh=bbox_thresh,
                                            nms_thresh=nms_thresh,
                                            kernel=kernel,
                                            img_path=img)

        compute_time_agg += time.time() - new_frame_time

        if frame_id == 1 or detections is None:
            sahi_obj_pred_list.append((frame_id - 1, ()))
        else:
            if return_sahi_obj_pred_list:
                sahi_obj_pred_list.append((frame_id - 1, [ObjectPrediction(
                    bbox=det,
                    category_id=0,
                    score=0.6,
                    bool_mask=None,
                    category_name="bird",
                    shift_amount=[0, 0],
                    full_shape=None,
                ) for det in detections]))

            # Collect MOT data
            for det in detections:
                new_row = [
                    int(frame_id),
                    -1,
                    det[0],
                    det[1],
                    det[2],
                    det[3],
                    float(0.6)
                ]
                if np.shape(mot_matrix_preds)[0] == 0:
                    mot_matrix_preds = new_row
                else:
                    mot_matrix_preds = np.vstack((mot_matrix_preds, new_row))

    if output_path:
        output_path.mkdir(exist_ok=True, parents=True)
        np.savetxt(output_path / f"{sequence_name}.txt", mot_matrix_preds, delimiter=",")

    result = {
        "timing": {
            "compute_time": compute_time_agg,
            "frame_count": frame_id
        }
    }
    if return_sahi_obj_pred_list:
        result["sahi_obj_pred_list"] = sahi_obj_pred_list

    return result
