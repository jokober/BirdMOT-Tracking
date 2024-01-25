import json
import pickle
from argparse import ArgumentParser
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--sahi_predictions_params_path", type=Path, required=True)
    parser.add_argument("--device", type=str, required=False, default='cpu')
    parser.add_argument("--out_path", type=Path, required=True)

    args = parser.parse_args()

    assert args.sahi_predictions_params_path, f"Prediction parameter file {args.sahi_predictions_params_path} does not exist"
    with open(args.sahi_predictions_params_path) as json_file:
        predictions_params = json.load(json_file)


    p = Path(args.source)
    for child in p.glob('**/images/'):
        seq_name = child.parent.name
        print(seq_name)

        sahi_prediction_on_folder(child, args.model_path, predictions_params, args.device, args.out_path, name=seq_name)
