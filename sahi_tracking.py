from argparse import ArgumentParser
from pathlib import Path, PosixPath
from typing import Union

from sahi.utils.coco import Coco


def sahi_tracking_from_predictions(predictions: Union[Path, Coco]):
    if type(predictions) == PosixPath:
        predictions = Coco.from_coco_dict_or_path(predictions.as_posix())

    predicitons = [image.predictions for image in predictions]
    print(predictions)

def predict():
#class SahiTracking:
#    def __init__(self, tracker, tracker_args):
 #       if tracker == "ByteTrack":
 #           self.tracker = ByteTrack(**tracker_args)

#    def update(self):
#        pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predictions_path", type=Path, required=True)
    parser.add_argument("--device", type=str, required=False, default='cpu')
    args = parser.parse_args()

    sahi_tracking_from_predictions(args.predictions_path)