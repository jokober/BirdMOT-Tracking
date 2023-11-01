from copy import deepcopy
from pathlib import Path

import numpy as np
from deepdiff import DeepHash

from src.sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from src.sahi_tracking.formats.mot_format import create_mot_folder_structure
from src.sahi_tracking.helper.config import get_predictions_path, get_tracking_results_path
from src.sahi_tracking.trackers.norfair_tracker import NorfairTracker
from src.sahi_tracking.trackers.oc_sort_tracker import OC_Sort_Tracker


def find_or_create_tracking_results(tracking_experiment: dict, predictions_result: dict, dataset: dict, persistence_state: DataStatePersistance, overwrite_existing: bool = False):
    predictions_result = deepcopy(predictions_result)
    tracking_experiment = deepcopy(tracking_experiment)

    tracking_results = {
        'dataset_hash': dataset['hash'],
        'predictions_hash': predictions_result['hash'],
        'tracking_experiment': tracking_experiment,
        'tracking_results': {},
        'hash': None
    }

    # Create hash of the tracking results
    deephash_exclude_paths = [
        "root['tracking_results']",
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
        for sequence in predictions_result['predictions']:
            #mot_path, mot_img_path, det_path, gt_path = create_mot_folder_structure(sequence['seq_name'], tracking_results_path / 'MOT')

            # Prepare Tracker
            if tracking_experiment['tracker_type'] == 'noirfair':
                tracker = NorfairTracker(accumulate_results=True, **tracking_experiment['tracker_config'])
            elif tracking_experiment['tracker_type'] == 'oc_sort':
                tracker = OC_Sort_Tracker(accumulate_results=True, **tracking_experiment['tracker_config'])
            else:
                raise NotImplementedError("The tracker_type is not implemented.")

            for frame_pred in sorted(sequence['frame_predictions'], key=lambda d: d[0]):
                tracker.update_online(frame_pred[1])

            #tracking_results['tracking_results']['sequence'][sequence['seq_name']] = tracker.get_mot_list()


            # Save tracking results
            motformat_result_path =tracking_results_path / f"{dataset['dataset_config']['benchmark_name']}-all" / "default_tracker" / "data"
            motformat_result_path.mkdir(exist_ok=True, parents=True)
            tracking_results['tracking_results']['result_path'] = motformat_result_path.parents[2]
            np.savetxt(motformat_result_path/ f"{sequence['seq_name']}.txt", tracker.get_mot_list(), delimiter=",")
            if sequence['seq_name'] == 'C0085_135558_135565':
                print("debug")

        tracking_results['hash'] = tracking_results_hash
        persistence_state.update_state('append', 'tracking_results', tracking_results)
    else:
        tracking_results = persistence_state.load_data('tracking_results', tracking_results_hash)


    return tracking_results