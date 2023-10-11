import argparse
import json
from argparse import ArgumentParser
from pathlib import Path



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tracking_dataset_path", type=Path, required=True)
    parser.add_argument("--tracking_experiment_path", type=Path, required=True)
    parser.add_argument("--caching",  default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite_existing",  default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", type=str, required=False, default='cpu')
    args = parser.parse_args()



    tracking_dataset_path = args.tracking_dataset.resolve()
    assert tracking_dataset_path, f"Experiment config {tracking_dataset_path} does not exist"
    with open(tracking_dataset_path) as json_file:
        tracking_dataset_dict = json.load(json_file)

    if not args.cache:
        predictions = Coco.from_coco_dict_or_path(predictions.as_posix())

    predicitons = [image.predictions for image in predictions]
    print(predictions)