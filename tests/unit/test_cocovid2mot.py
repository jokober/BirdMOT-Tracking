from src.sahi_tracking.formats.cocovid2mot import cocovid2mot
from tests.fixtures.fixtures import coco_annotations_fixture_dir, cocovid_images_fixture_dir, primary_cocovid_fixture_path, \
    primary_sequence_dict_fixture


def test_cocovid2mot(tmp_path, primary_sequence_dict_fixture):
    sequences = cocovid2mot(primary_sequence_dict_fixture, tmp_path, cocovid_images_fixture_dir)
    sequences[0]['mot_path'].exists()