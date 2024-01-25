import json
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sahi.predict import predict

from sahi_tracking.formats.mot_format import mot_matrix_from_sahi_object_prediction_list
import cv2 as cv
def backsub_det(input, algo, bs_config):
    if algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2(varThreshold=bs_config.var_thresh, detectShadows=True)
        backSub.setShadowThreshold(bs_config.shadow_thresh)

    else:
        backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=bs_config.dist2Threshold, detectShadows=True)

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

    while True:
        ret, frame_bgr = capture.read()
        if frame_bgr is None:
            break

        # time when we finish processing for this frame
        new_frame_time = time.time()

        # get detections
        detections = get_detections_backsub(backSub,
                                    cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY),
                                    bbox_thresh=bs_config.bbox_thresh,
                                    nms_thresh=bs_config.nms_thresh,
                                    kernel=bs_config.kernel)

        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        if detections is not None:
            draw_bboxes(frame_bgr, detections)

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        print(fps)

        if True:
            cv.imshow('frame_bgr', frame_bgr)

            keyboard = cv.waitKey(1)
            if keyboard == 'q' or keyboard == 27:
                break

    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

def sahi_prediction_on_folder(source, model_path: Path, prediction_params: dict, device, out_path: Path, name:str) -> np.ndarray:
    """
    Run sahi prediction on a folder and save the results in MOT format

    :param source: Folder with images
    :param model_path: Path to the model
    :param prediction_params: Sahi prediction parameters as dict
    :param device: Device to run the prediction on
    :param out_path: Path to save the results
    :param name: Name of the sequence
    :return:
    """
    # Run  batch prediction
    predict_return_dict = predict(
        source=source.as_posix(),
        model_path=model_path.as_posix(),
        model_device=device,
        project=out_path.as_posix(),
        name=name,
        return_dict=True,
        export_pickle=True,
        **prediction_params
    )
    # Load results for each image stem
    sahi_object_prediction_list = []
    for pickle_path in (Path(predict_return_dict['export_dir']) / 'pickles').glob('*.pickle'):
        print(pickle_path)
        frame_id = int(pickle_path.stem.split("-")[-1])
        with open(pickle_path, 'rb') as fp:
            res = pickle.load(fp)
        sahi_object_prediction_list.append((frame_id, res))

    # Create MOT format
    mot_matrix_pred = mot_matrix_from_sahi_object_prediction_list(name, sahi_object_prediction_list,
                                                output_path = out_path / f"{name}/det")

    return mot_matrix_pred

def run_bs_track_eval_pipeline():
    bs_config={
        "var_thresh": 16,
    "shadow_thresh": 0.5,
    "dist2Threshold": 1000,
    "bbox_thresh": 100,
    "nms_thresh": 1e-2,
    "kernel": np.array((9,9), dtype=np.uint8)
    }


    p = Path("path")
    for child in p.glob('**/images/'):
        seq_name = child.parent.name
        print(seq_name)

        sahi_prediction_on_folder(child, args.model_path, predictions_params, args.device, args.out_path, name=seq_name)

