from copy import deepcopy
from pathlib import Path

import numpy as np
from deepdiff import DeepHash
import matplotlib

from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from sahi_tracking.formats.mot_format import create_mot_folder_structure
from sahi_tracking.helper.config import get_predictions_path, get_tracking_results_path
from sahi_tracking.trackers.ByteTrack import ByteTrack
from sahi_tracking.trackers.norfair_tracker import NorfairTracker
from sahi_tracking.trackers.ocsorttracker import OcSortTracker
from sahi_tracking.trackers.sort_tracker import SORTTracker
from sahi_tracking.trackers.viou_tracker import VIoUTracker


def find_or_create_tracking_results(tracking_experiment: dict, predictions_result: dict, dataset: dict, persistence_state: DataStatePersistance, overwrite_existing: bool = False):
    predictions_result = deepcopy(predictions_result)
    tracking_experiment = deepcopy(tracking_experiment)

    tracking_results = {
        'dataset_hash': dataset['hash'],
        'predictions_hash': predictions_result['hash'],
        'tracking_experiment': tracking_experiment,
        'tracking_results': {},
        'name': None,
        'hash': None
    }

    # Create hash of the tracking results
    deephash_exclude_paths = [
        "root['tracking_results']",
        "root['name']"
        "root['hash']",
    ]
    tracking_results_hash = DeepHash(tracking_results, exclude_paths=deephash_exclude_paths)[tracking_results]

    # Delete existing predictions if overwrite_existing is True
    if overwrite_existing:
        persistence_state.delete_existing('tracking_results', tracking_results_hash)

    # Check if tracking results already exist. Return it or create otherwise.
    if not persistence_state.data_exists('tracking_results', tracking_results_hash):
        tracking_results_path = get_tracking_results_path() / tracking_results_hash
        tracking_results_path.mkdir(exist_ok=True)

        # Run tracking
        assert len(predictions_result['predictions']) == len(dataset['dataset']["sequences"])
        for sequence_preds, seq_info in zip(predictions_result['predictions'], dataset['dataset']["sequences"]):
            #mot_path, mot_img_path, det_path, gt_path = create_mot_folder_structure(sequence['seq_name'], tracking_results_path / 'MOT')

            # Prepare Tracker
            if tracking_experiment['tracker_type'] == 'noirfair':
                tracker = NorfairTracker(accumulate_results=True, **tracking_experiment['tracker_config'])
            elif tracking_experiment['tracker_type'] == 'oc_sort':
                tracker = OcSortTracker(accumulate_results=True, **tracking_experiment['tracker_config'])
            elif tracking_experiment['tracker_type'] == 'sort':
                tracker = SORTTracker(accumulate_results=True, **tracking_experiment['tracker_config'])
            elif tracking_experiment['tracker_type'] == 'viou':
                tracker = VIoUTracker(img_path=seq_info['mot_path'] / 'img1', **tracking_experiment['tracker_config'])
            elif tracking_experiment['tracker_type'] == 'bytetrack':
                tracker = ByteTrack(accumulate_results=True, args = tracking_experiment['tracker_config'])
            else:
                raise NotImplementedError("The tracker_type is not implemented.")

            for frame_pred in sorted(sequence_preds['frame_predictions'], key=lambda d: d[0]):
                tracker.update_online(frame_pred[1])

            if tracking_experiment['tracker_type'] == 'viou':
                tracker.collect_results()
            #tracking_results['tracking_results']['sequence'][sequence['seq_name']] = tracker.get_mot_list()


            # Save tracking results
            motformat_result_path =tracking_results_path / f"{dataset['dataset_config']['benchmark_name']}-all" / "default_tracker" / "data"
            motformat_result_path.mkdir(exist_ok=True, parents=True)
            tracking_results['tracking_results']['result_data_path'] = motformat_result_path
            tracking_results['tracking_results']['result_path'] = motformat_result_path.parents[2]
            np.savetxt(motformat_result_path/ f"{sequence_preds['seq_name']}.txt", tracker.get_mot_list(), delimiter=",")

        tracking_results['name'] = tracker.name
        tracking_results['hash'] = tracking_results_hash
        persistence_state.update_state('append', 'tracking_results', tracking_results)
    else:
        tracking_results = persistence_state.load_data('tracking_results', tracking_results_hash)


    return tracking_results