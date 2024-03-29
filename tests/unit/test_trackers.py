from sahi_tracking.experiments_framework.dataset_creation import find_or_create_dataset
from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from sahi_tracking.experiments_framework.evaluation_result_creation import find_or_create_tracker_evaluations
from sahi_tracking.experiments_framework.predictions_creation import find_or_create_predictions
from sahi_tracking.experiments_framework.tracking_result_creation import find_or_create_tracking_results
from sahi_tracking.helper.config import get_datasets_path
from tests.fixtures.fixtures import dataset_config_dict_fixture, cocovid_images_fixture_dir, \
    sahi_prediction_params_dict_fixture, sahi_test_model_path, tracking_experiments_dict_fixture, \
    tracking_viou_experiment_dict_fixture, tracking_oc_sort_experiment_dict_fixture, \
    tracking_bytetrack_experiment_dict_fixture, tracking_sort_experiment_dict_fixture


def test_viou_tracker(tmp_path, dataset_config_dict_fixture, sahi_prediction_params_dict_fixture,
                      tracking_viou_experiment_dict_fixture):
    persistence_state = DataStatePersistance()

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

    for multiplier, one_experiment in enumerate(tracking_viou_experiment_dict_fixture['tracker_experiments'], start=1):
        tracking_results = find_or_create_tracking_results(tracking_experiment=one_experiment,
                                                           predictions_result=predictions_result,
                                                           dataset=dataset,
                                                           persistence_state=persistence_state,
                                                           overwrite_existing=False)

        evaluation_results = find_or_create_tracker_evaluations(tracking_results=tracking_results,
                                                                predictions_result=predictions_result,
                                                                dataset=dataset,
                                                                persistence_state=persistence_state,
                                                                overwrite_existing=True)
        print(evaluation_results)


def test_sort_tracker(tmp_path, dataset_config_dict_fixture, sahi_prediction_params_dict_fixture,
                      tracking_sort_experiment_dict_fixture):
    persistence_state = DataStatePersistance()

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

    for multiplier, one_experiment in enumerate(tracking_sort_experiment_dict_fixture['tracker_experiments'], start=1):
        tracking_results = find_or_create_tracking_results(tracking_experiment=one_experiment,
                                                           predictions_result=predictions_result,
                                                           dataset=dataset,
                                                           persistence_state=persistence_state,
                                                           overwrite_existing=False)

        evaluation_results = find_or_create_tracker_evaluations(tracking_results=tracking_results,
                                                                predictions_result=predictions_result,
                                                                dataset=dataset,
                                                                persistence_state=persistence_state,
                                                                overwrite_existing=True)

def test_oc_sort_tracker(tmp_path, dataset_config_dict_fixture, sahi_prediction_params_dict_fixture,
                         tracking_oc_sort_experiment_dict_fixture):
    persistence_state = DataStatePersistance()

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

    for multiplier, one_experiment in enumerate(tracking_oc_sort_experiment_dict_fixture['tracker_experiments'],
                                                start=1):
        tracking_results = find_or_create_tracking_results(tracking_experiment=one_experiment,
                                                           predictions_result=predictions_result,
                                                           dataset=dataset,
                                                           persistence_state=persistence_state,
                                                           overwrite_existing=False)

        evaluation_results = find_or_create_tracker_evaluations(tracking_results=tracking_results,
                                                                predictions_result=predictions_result,
                                                                dataset=dataset,
                                                                persistence_state=persistence_state,
                                                                overwrite_existing=True)
        print(evaluation_results)


def test_bytetrack_tracker(tmp_path, dataset_config_dict_fixture, sahi_prediction_params_dict_fixture,
                           tracking_bytetrack_experiment_dict_fixture):
    persistence_state = DataStatePersistance()

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

    for multiplier, one_experiment in enumerate(tracking_bytetrack_experiment_dict_fixture['tracker_experiments'],
                                                start=1):
        tracking_results = find_or_create_tracking_results(tracking_experiment=one_experiment,
                                                           predictions_result=predictions_result,
                                                           dataset=dataset,
                                                           persistence_state=persistence_state,
                                                           overwrite_existing=False)

        evaluation_results = find_or_create_tracker_evaluations(tracking_results=tracking_results,
                                                                predictions_result=predictions_result,
                                                                dataset=dataset,
                                                                persistence_state=persistence_state,
                                                                overwrite_existing=True)
        print(evaluation_results)
