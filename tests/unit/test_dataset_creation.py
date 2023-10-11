from src.sahi_tracking.experiments_framework.dataset_creation import find_or_create_dataset
from src.sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from src.sahi_tracking.helper.config import get_datasets_path
from tests.fixtures.fixtures import dataset_config_dict_fixture, cocovid_images_fixture_dir


def test_dataset_creation(tmp_path, dataset_config_dict_fixture):
    persistence_state = DataStatePersistance()
    find_or_create_dataset(dataset_config_dict_fixture, persistence_state, cocovid_images_fixture_dir, overwrite_existing = False)
    assert len(persistence_state.state['datasets']) == 1

    find_or_create_dataset(dataset_config_dict_fixture, persistence_state, cocovid_images_fixture_dir, overwrite_existing=False)
    assert len(persistence_state.state['datasets']) == 1