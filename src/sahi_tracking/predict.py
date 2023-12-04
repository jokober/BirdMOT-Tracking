import pickle
from pathlib import Path

import numpy as np
from sahi.predict import predict

from sahi_tracking.formats.mot_format import mot_matrix_from_sahi_object_prediction_list


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
        source=source,
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
        frame_id = int(pickle_path.stem)
        with open(pickle_path, 'rb') as fp:
            res = pickle.load(fp)
        sahi_object_prediction_list.append((frame_id, res))

    # Create MOT format
    mot_matrix_pred = mot_matrix_from_sahi_object_prediction_list(name, sahi_object_prediction_list,
                                                out_path / f"{name}/MOT/det")

    return mot_matrix_pred