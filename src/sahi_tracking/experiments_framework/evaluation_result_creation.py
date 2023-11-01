from copy import deepcopy
from pathlib import Path

import numpy as np
from deepdiff import DeepHash

from src.sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from src.sahi_tracking.formats.mot_format import create_mot_folder_structure
from src.sahi_tracking.helper.config import get_predictions_path, get_tracking_results_path, get_evaluation_results_path
from src.sahi_tracking.trackers.norfair_tracker import NorfairTracker
from src.sahi_tracking.trackeval_evaluation import trackeval_evaluate



def find_or_create_tracker_evaluations(tracking_results: dict, predictions_result: dict, dataset: dict,
                                       persistence_state: DataStatePersistance, overwrite_existing: bool = False):
    predictions_result = deepcopy(predictions_result)
    tracking_experiment = deepcopy(tracking_results)

    evaluation_results = {
        'dataset_hash': dataset['hash'],
        'predictions_hash': predictions_result['hash'],
        'tracking_results_hash': tracking_experiment['hash'],
        'evaluation_results': None,
        'hash': None
    }

    # Create hash of the tracking results
    deephash_exclude_paths = [
        "root['evaluation_results']",
        "root['hash']",
    ]
    evaluation_results_hash = DeepHash(evaluation_results, exclude_paths=deephash_exclude_paths)[evaluation_results]

    # Delete existing predictions if overwrite_existing is True
    if overwrite_existing:
        persistence_state.delete_existing('evaluation_results', evaluation_results_hash)

    # Check if tracking results already exist. Return it or create otherwise.
    if not persistence_state.data_exists('evaluation_results', evaluation_results_hash):
        evaluation_results_path = get_evaluation_results_path() / evaluation_results_hash
        evaluation_results_path.mkdir(exist_ok=True, parents=True)

        evaluation_results['evaluation_results'] = trackeval_evaluate(
            GT_FOLDER=dataset['dataset']['dataset_path'] / 'MOT',
            TRACKERS_FOLDER=tracking_results['tracking_results']['result_path'],
            OUTPUT_FOLDER=evaluation_results_path,
            SEQMAP_FILE=dataset['dataset']['seq_map'],
            BENCHMARK=dataset['dataset_config']['benchmark_name'],
            SPLIT_TO_EVAL="all"
            )

        evaluation_results['hash'] = evaluation_results_hash
        persistence_state.update_state('append', 'evaluation_results', evaluation_results)
    else:
        evaluation_results = persistence_state.load_data('evaluation_results', evaluation_results_hash)

    return evaluation_results
