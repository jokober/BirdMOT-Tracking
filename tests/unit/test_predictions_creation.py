from src.sahi_tracking.experiments_framework.dataset_creation import find_or_create_dataset
from src.sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from src.sahi_tracking.experiments_framework.predictions_creation import find_or_create_predictions
from src.sahi_tracking.helper.config import get_datasets_path
from tests.fixtures.fixtures import dataset_config_dict_fixture, cocovid_images_fixture_dir, \
    sahi_prediction_params_dict_fixture, sahi_test_model_path


def test_predictions_creation(tmp_path, dataset_config_dict_fixture, sahi_prediction_params_dict_fixture):
    persistence_state = DataStatePersistance()

    dataset = find_or_create_dataset(dataset_config_dict_fixture,
                                     persistence_state=persistence_state,
                                     cocovid_img_path=cocovid_images_fixture_dir,
                                     overwrite_existing = False)
    find_or_create_predictions(dataset,
                               prediction_params=sahi_prediction_params_dict_fixture,
                               model_path=sahi_test_model_path,
                               persistence_state=persistence_state,
                               device='cpu',
                               cocovid_img_path=cocovid_images_fixture_dir,
                               overwrite_existing=True)
    assert len(persistence_state.state['datasets']) == 1

    find_or_create_predictions(dataset,
                               prediction_params=sahi_prediction_params_dict_fixture,
                               model_path=sahi_test_model_path,
                               persistence_state=persistence_state,
                               device='cpu',
                               cocovid_img_path=cocovid_images_fixture_dir,
                               overwrite_existing=True)
    assert len(persistence_state.state['datasets']) == 1