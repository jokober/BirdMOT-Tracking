from sahi_tracking.evaluation.trackeval_utils import get_linspaced_metric_names, trackeval_to_pandas
from tests.fixtures.fixtures import evaluation_result_fixture

def test_trackeval_to_pandas(evaluation_result_fixture):
   trackeval_to_pandas(evaluation_result_fixture['evaluation_results'][0]['MotChallenge2DBox']['default_tracker'])


def test_get_linspaced_metric_names():
    linspace_metric_strings = get_linspaced_metric_names('HOTA')
    for mn in ['HOTA___5',
               'HOTA___10',
               'HOTA___15',
               'HOTA___20',
               'HOTA___25',
               'HOTA___30',
               'HOTA___35',
               'HOTA___40',
               'HOTA___45',
               'HOTA___50',
               'HOTA___55',
               'HOTA___60',
               'HOTA___65',
               'HOTA___70',
               'HOTA___75',
               'HOTA___80',
               'HOTA___85',
               'HOTA___90',
               'HOTA___95']:
        assert mn in linspace_metric_strings


