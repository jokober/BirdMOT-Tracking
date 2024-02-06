from pathlib import Path

import streamlit as st

from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from sahi_tracking.experiments_framework.dataset_creation import find_or_create_dataset
from sahi_tracking.experiments_framework.evaluation_result_creation import find_or_create_tracker_evaluations
from sahi_tracking.experiments_framework.predictions_creation import find_or_create_predictions
from sahi_tracking.experiments_framework.tracking_result_creation import find_or_create_tracking_results
from sahi_tracking.experiments_framework.utils import load_config_files
from sahi_tracking.helper.config import get_local_data_path


@st.cache_data
def get_tracking_experiment_conf(tracking_hash: str):
    persistence_state = DataStatePersistance()
    return persistence_state.load_data('tracking_results', tracking_hash)


@st.cache_data
def load_data(tracking_dataset_dict, sahi_predictions_params_dict, tracking_experiment_dict):
    persistence_state = DataStatePersistance()
    dataset = find_or_create_dataset(tracking_dataset_dict,
                                     persistence_state=persistence_state,
                                     cocovid_img_path=get_local_data_path() / "dataset/cocovid/images",
                                     overwrite_existing=False)

    predictions_result = find_or_create_predictions(dataset,
                                                    prediction_params=sahi_predictions_params_dict,
                                                    model_path=st.session_state['model_path'],
                                                    persistence_state=persistence_state,
                                                    device="cpu",
                                                    cocovid_img_path=get_local_data_path() / "dataset/cocovid/images",
                                                    overwrite_existing=False)

    evaluation_results_list = []
    for one_experiment in tracking_experiment_dict['tracker_experiments']:
        tracking_results = find_or_create_tracking_results(tracking_experiment=one_experiment,
                                                           predictions_result=predictions_result,
                                                           dataset=dataset,
                                                           persistence_state=persistence_state,
                                                           overwrite_existing=False)

        evaluation_results = find_or_create_tracker_evaluations(tracking_results=tracking_results,
                                                                predictions_result=predictions_result,
                                                                dataset=dataset,
                                                                persistence_state=persistence_state,
                                                                overwrite_existing=False)

        evaluation_results_list.append(evaluation_results)

    return persistence_state, dataset, predictions_result, evaluation_results_list


@st.cache_data
def load_datafrom_state(tracking_dataset_dict, predictions_params_dict, tracking_experiment_dict):
    pass
