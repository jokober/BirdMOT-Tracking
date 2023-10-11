import json
from pathlib import Path

import pytest

# CocoVID
coco_annotations_fixture_dir = (Path(__file__).parents[1] / 'fixtures' / 'local_data/dataset/cocovid/coco_files')
cocovid_images_fixture_dir = (Path(__file__).parents[1] / 'fixtures' / 'local_data/dataset/cocovid/images')
primary_cocovid_fixture_path = coco_annotations_fixture_dir / 'C0054_783015_783046_scalabel_converted_coco_format_track_box.json'

# Config
primary_dataset_config_fixture_path = (Path(__file__).parents[1] / 'fixtures' / 'local_data/config/tracking_dataset/test_dataset_config.json')
sahi_prediction_params_fixture_path = (Path(__file__).parents[1] / 'fixtures' / 'local_data/config/sahi_prediction_params/test_prediction_params.json')
sahi_test_model_path = (Path(__file__).parents[1] / 'fixtures' / 'test_model_weights.pt')

@pytest.fixture
def dataset_config_dict_fixture():
    with open(primary_dataset_config_fixture_path) as json_file:
        return json.load(json_file)

@pytest.fixture
def primary_sequence_dict_fixture():
    with open(primary_cocovid_fixture_path) as json_file:
        return json.load(json_file)

@pytest.fixture
def sahi_prediction_params_dict_fixture():
    with open(sahi_prediction_params_fixture_path) as json_file:
        return json.load(json_file)