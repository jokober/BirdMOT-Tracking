from copy import deepcopy

from sahi_tracking.experiments_framework.dataset_creation import find_or_create_dataset
from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from sahi_tracking.experiments_framework.predictions_creation import find_or_create_predictions
from sahi_tracking.experiments_framework.tracking_result_creation import find_or_create_tracking_results
from sahi_tracking.helper.config import get_datasets_path
from tests.fixtures.fixtures import dataset_config_dict_fixture, cocovid_images_fixture_dir, \
    sahi_prediction_params_dict_fixture, sahi_test_model_path, tracking_experiments_dict_fixture


def test_find_or_create_tracking_results(tmp_path, dataset_config_dict_fixture, sahi_prediction_params_dict_fixture,
                                         tracking_experiments_dict_fixture):
    persistence_state = DataStatePersistance()
    persistence_state.create_new_state()

    dataset = find_or_create_dataset(dataset_config_dict_fixture,
                                     persistence_state=persistence_state,
                                     cocovid_img_path=cocovid_images_fixture_dir,
                                     overwrite_existing=False)
    predictions_result = find_or_create_predictions(dataset,
                                                    prediction_params=sahi_prediction_params_dict_fixture,
                                                    model_path=sahi_test_model_path,
                                                    persistence_state=persistence_state,
                                                    device='cpu',
                                                    cocovid_img_path=cocovid_images_fixture_dir,
                                                    overwrite_existing=True)

    hashes = []
    for multiplier, one_experiment in enumerate(tracking_experiments_dict_fixture['tracker_experiments'], start=0):
        tracking_results = find_or_create_tracking_results(tracking_experiment=one_experiment,
                                        predictions_result=predictions_result,
                                        dataset=dataset,
                                        persistence_state=persistence_state,
                                        overwrite_existing=False)
        hashes.append(tracking_results['hash'])
        assert len(persistence_state.state['tracking_results']) == 1 + (2* multiplier)

        tracking_results = find_or_create_tracking_results(tracking_experiment=one_experiment,
                                        predictions_result=predictions_result,
                                        dataset=dataset,
                                        persistence_state=persistence_state,
                                        overwrite_existing=False)
        hashes.append(tracking_results['hash'])
        assert len(persistence_state.state['tracking_results']) ==  1 + (2* multiplier)

        modified_exp = deepcopy(one_experiment)
        modified_exp['NonsenseKey'] = 'NonsenseValue'
        tracking_results = find_or_create_tracking_results(tracking_experiment=modified_exp,
                                        predictions_result=predictions_result,
                                        dataset=dataset,
                                        persistence_state=persistence_state,
                                        overwrite_existing=False)
        hashes.append(tracking_results['hash'])
        assert len(persistence_state.state['tracking_results']) ==  2 + (2* multiplier)



    assert len(set(hashes)) == len(persistence_state.state['tracking_results'])

