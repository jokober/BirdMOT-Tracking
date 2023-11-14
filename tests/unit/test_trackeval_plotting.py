import pytest

from conftest import ValueStorage
from sahi_tracking.evaluation.trackeval_plotting import load_trackeval_evaluation_data, plot_compare_trackers
from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from tests.fixtures.fixtures import evaluation_result_fixture

def test_load_trackeval_evaluation_data(evaluation_result_fixture):
    load_trackeval_evaluation_data(ValueStorage.state_persistance.state['evaluation_results'], "pedestrian")


def test_plotting_eval():
    data = load_trackeval_evaluation_data(ValueStorage.state_persistance.state['evaluation_results'], "pedestrian")
    plot_compare_trackers(data, "pedestrian", output_folder = ".", plots_list= None)
