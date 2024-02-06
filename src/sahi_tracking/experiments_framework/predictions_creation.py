import pickle
import time
from copy import deepcopy
from pathlib import Path

from deepdiff import DeepHash
from sahi.predict import predict

from sahi_tracking.background_substraction.backsub_mot import backsub_on_mot_data
from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from sahi_tracking.formats.mot_format import mot_matrix_from_sahi_object_prediction_list
from sahi_tracking.helper.config import get_predictions_path


def find_or_create_predictions(dataset: dict, prediction_params: dict, model_path: Path,
                               persistence_state: DataStatePersistance, device='cpu', cocovid_img_path: Path = None,
                               overwrite_existing: bool = False):
    dataset = deepcopy(dataset)
    prediction_params = deepcopy(prediction_params)

    predictions_results = {
        'prediction_params': prediction_params,
        'dataset_hash': dataset['hash'],
        'dir': None,
        'predictions': [],
        'fps': None,
        'hash': None

    }

    # Create hash of the predictions only based on the predictions config
    deephash_exclude_paths = [
        "root['predictions']",
        "root['dir']",
        "root['fps']",
        "root['hash']",
    ]
    predictions_results_hash = DeepHash(predictions_results, exclude_paths=deephash_exclude_paths)[predictions_results]

    # Delete existing predictions if overwrite_existing is True
    if overwrite_existing:
        persistence_state.delete_existing_by_hash('predictions_results', predictions_results_hash)

    # Check if dataset already exists return it if so or create otherwise
    if not persistence_state.data_exists('predictions_results', predictions_results_hash):
        prediction_results_path = get_predictions_path() / predictions_results_hash
        prediction_results_path.mkdir(exist_ok=True, parents=True)

        # import time
        # print("time")
        # time.sleep(30)
        if "detector_type" not in prediction_params or prediction_params["detector_type"] == "sahi":
            assert "slice_height" in prediction_params
            del prediction_params["detector_type"]
            compute_time = 0
            frame_count = 0
            for sequence in dataset['dataset']['sequences']:
                start_timer = time.time()

                # Run  batch prediction
                predict_return_dict = predict(
                    source=sequence['mot_path'] / 'img1',
                    model_path=model_path.as_posix(),
                    model_device=0,  # ToDo:revert device,
                    project=prediction_results_path.as_posix(),
                    name=sequence['name'],
                    return_dict=True,
                    export_pickle=True,
                    **prediction_params
                )
                compute_time += time.time() - start_timer
                # Load results for each image stem
                seq_res = {
                    'seq_name': sequence['name'],
                    'frame_predictions': [],
                }
                for pickle_path in (Path(predict_return_dict['export_dir']) / 'pickles').glob('*.pickle'):
                    frame_id = int(pickle_path.stem)
                    with open(pickle_path, 'rb') as fp:
                        res = pickle.load(fp)
                    seq_res['frame_predictions'].append((frame_id, res))
                predictions_results['predictions'].append(seq_res)

                frame_count += len(seq_res['frame_predictions'])

                # Create MOT format
                mot_matrix_from_sahi_object_prediction_list(seq_res['seq_name'], seq_res['frame_predictions'],
                                                            prediction_results_path / f"{sequence['name']}/MOT/det")

        elif prediction_params["detector_type"] == "background_substraction":
            compute_time = 0
            frame_count = 0

            for sequence in dataset['dataset']['sequences']:
                print(sequence['name'])
                assert "history" in prediction_params
                # Creat MOT data and sahi prediction list as the trackers are only implemented for sahi data
                pred_res = backsub_on_mot_data(sequence_name=sequence['name'], mot_path=sequence['mot_path'],
                                               output_path=prediction_results_path / f"{sequence['name']}/MOT/det",
                                               return_sahi_obj_pred_list=True, **prediction_params)
                compute_time += pred_res['timing']["compute_time"]
                frame_count += pred_res['timing']["frame_count"]

                seq_res = {
                    'seq_name': sequence['name'],
                    'frame_predictions': pred_res["sahi_obj_pred_list"],
                }
                predictions_results['predictions'].append(seq_res)

        predictions_results['fps'] = frame_count / compute_time
        predictions_results['dir'] = prediction_results_path
        predictions_results['hash'] = predictions_results_hash

        # Add new predictions to state
        persistence_state.update_state('append', 'predictions_results', predictions_results)
    else:
        predictions_results = persistence_state.load_data('predictions_results', predictions_results_hash)

    return predictions_results
