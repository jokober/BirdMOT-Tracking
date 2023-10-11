from copy import deepcopy
from pathlib import Path

from deepdiff import DeepHash
from sahi.predict import predict

from src.sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from src.sahi_tracking.formats.mot_format import create_mot_folder_structure
from src.sahi_tracking.helper.config import get_predictions_path


def find_or_create_predictions(dataset: dict, prediction_params: dict, model_path: Path, persistence_state: DataStatePersistance, device = 'cpu', cocovid_img_path: Path = None, overwrite_existing: bool = False):
    dataset = deepcopy(dataset)
    prediction_params = deepcopy(prediction_params)

    predictions_results = {
        'prediction_params': prediction_params,
        'dataset_hash': dataset['hash'],
        'predictions': None,
        'hash': None
    }

    # Create hash of the predictions only based on the predictions config
    deephash_exclude_paths = [
        "root['predictions']",
        "root['hash']",
    ]
    predictions_results_hash = DeepHash(predictions_results, exclude_paths=deephash_exclude_paths)[predictions_results]

    # Delete existing predictions if overwrite_existing is True
    if overwrite_existing:
        persistence_state.delete_existing('predictions_results', predictions_results_hash)

    # Check if dataset already exists return it if so or create otherwise
    if not persistence_state.data_exists('predictions_results', predictions_results_hash):
        prediction_results_path = get_predictions_path() / predictions_results_hash
        prediction_results_path.mkdir(exist_ok=True)
        for sequence in dataset['dataset']['sequences']:
            create_mot_folder_structure(sequence['name'], prediction_results_path / 'MOT')

            predictions = predict(
                source=sequence['mot_path'] / 'img1',
                model_path=model_path.as_posix(),
                model_device=device,
                project=prediction_results_path.as_posix(),
                name=sequence['name'],
                **prediction_params
            )
            predictions_results['predictions'] = predictions
            predictions_results['hash'] = predictions_results_hash

            # Add new predictions to state
            persistence_state.update_state('append', 'predictions_results', predictions_results)
        else:
            predictions_results = persistence_state.load_data('predictions_results', predictions_results_hash)


    return predictions_results