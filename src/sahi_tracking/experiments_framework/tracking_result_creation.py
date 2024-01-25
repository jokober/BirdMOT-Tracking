import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from deepdiff import DeepHash
import matplotlib
from yupi.core.featurizers import DistanceFeaturizer, DisplacementFeaturizer

from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from sahi_tracking.formats.mot_format import create_mot_folder_structure
from sahi_tracking.helper.config import get_predictions_path, get_tracking_results_path
from sahi_tracking.trackers.ByteTrack import ByteTrack
from sahi_tracking.trackers.norfair_tracker import NorfairTracker
from sahi_tracking.trackers.ocsorttracker import OcSortTracker
from sahi_tracking.trackers.sort_tracker import SORTTracker
from sahi_tracking.trackers.viou_tracker import VIoUTracker
from sahi_tracking.trajectories.from_mot import yupi_traj_from_mot


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
        persistence_state.delete_existing_by_hash('tracking_results', tracking_results_hash)

    # Check if tracking results already exist. Return it or create otherwise.
    if not persistence_state.data_exists('tracking_results', tracking_results_hash):
        tracking_results_path = get_tracking_results_path() / tracking_results_hash
        tracking_results_path.mkdir(exist_ok=True)

        # Run tracking
        assert len(predictions_result['predictions']) == len(dataset['dataset']["sequences"])
        frame_counter = 0
        time_aggregator = 0
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

            time_start = time.time()
            for frame_pred in sorted(sequence_preds['frame_predictions'], key=lambda d: d[0]):
                #print(frame_pred)
                frame_counter+=1
                tracker.update_online(frame_pred[1])

            time_aggregator += time.time() - time_start
            fps = frame_counter / time_aggregator

            if tracking_experiment['tracker_type'] == 'viou':
                tracker.collect_results()
            #tracking_results['tracking_results']['sequence'][sequence['seq_name']] = tracker.get_mot_list()

            # Get mot results as numpy
            mot_data = tracker.get_mot_list()

            # Filter Tracks by trajectory features
            if len(mot_data) > 0:
                if "filter" in tracking_experiment:
                    if tracking_experiment['filter']['filter_type'] == "simple":
                        instance_ids, trajs = yupi_traj_from_mot(mot_data)
                        assert len(instance_ids) == len(trajs)
                        if 'min_distance' in tracking_experiment['filter']['filter_config']:
                            distances = DistanceFeaturizer(0).featurize(trajs)
                            instance_ids = instance_ids.reshape(-1,1)
                            short_distance_instance_ids =instance_ids[distances[:,0] < tracking_experiment['filter']['filter_config']['min_distance']]

                        if 'min_displacement' in tracking_experiment['filter']['filter_config']:
                            displacements = DisplacementFeaturizer(0).featurize(trajs)
                            instance_ids = instance_ids.reshape(-1,1)
                            short_displacement_instance_ids =instance_ids[displacements[:,0] < tracking_experiment['filter']['filter_config']['min_displacement']]

                        if 'min_track_length' in tracking_experiment['filter']['filter_config']:
                            md=deepcopy(mot_data)
                            counts = np.array(np.unique(md[:,1], return_counts=True))
                            counts = np.reshape(np.ravel(counts), [-1, 2], order="F")
                            short_tracks_instance_ids =counts[counts[:,1] < tracking_experiment['filter']['filter_config']['min_track_length']][:,0]


                        for id in list(short_distance_instance_ids) + list(short_displacement_instance_ids) + list(short_tracks_instance_ids):
                            mot_data = mot_data[~(mot_data[:,1] == id) ]

            # Save tracking results
            motformat_result_path =tracking_results_path / f"{dataset['dataset_config']['benchmark_name']}-all" / "default_tracker" / "data"
            motformat_result_path.mkdir(exist_ok=True, parents=True)
            tracking_results['tracking_results']['result_data_path'] = motformat_result_path
            tracking_results['tracking_results']['fps'] = fps
            tracking_results['tracking_results']['result_path'] = motformat_result_path.parents[2]
            np.savetxt(motformat_result_path/ f"{sequence_preds['seq_name']}.txt", mot_data, delimiter=",")

        tracking_results['name'] = f"{tracker.name}+Filter" if "filter" in tracking_experiment else tracker.name
        tracking_results['hash'] = tracking_results_hash
        persistence_state.update_state('append', 'tracking_results', tracking_results)

    else:
        tracking_results = persistence_state.load_data('tracking_results', tracking_results_hash)


    return tracking_results