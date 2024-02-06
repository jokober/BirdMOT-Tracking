from typing import List

import pandas as pd

from sahi_tracking.evaluation.trackeval_evaluation import load_trackeval_evaluation_data

METRICS = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
LINSPACED_METRICS = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'AssRe', 'AssPr', 'AssA', 'LocA', 'DetRe', 'DetPr', 'DetA',
                     'HOTA']


def trackeval_to_pandas(trackeval_res: dict):
    results_dict = {}
    for key, value in trackeval_res.items():
        hota_res = value['pedestrian']['HOTA']
        results_dict[key] = [*[hota_res[m] for m in METRICS], *hota_res['HOTA_TP'],
                             *hota_res['HOTA_FN'], *hota_res['HOTA_FP'], *hota_res['AssRe'], *hota_res['AssPr'],
                             *hota_res['AssA'], *hota_res['LocA'], *hota_res['DetRe'], *hota_res['DetPr'],
                             *hota_res['DetA'], *hota_res['HOTA']]

    column_names = [*METRICS, *get_linspaced_metric_names('HOTA_TP'), *get_linspaced_metric_names('HOTA_FN'),
                    *get_linspaced_metric_names('HOTA_FP'), *get_linspaced_metric_names('AssRe'),
                    *get_linspaced_metric_names('AssPr'),
                    *get_linspaced_metric_names('AssA'), *get_linspaced_metric_names('LocA'),
                    *get_linspaced_metric_names('DetRe'),
                    *get_linspaced_metric_names('DetPr'), *get_linspaced_metric_names('DetA'),
                    *get_linspaced_metric_names('HOTA')]
    assert len(results_dict[key]) == len(column_names)

    return pd.DataFrame.from_dict(results_dict, orient='index',
                                  columns=column_names)


def evaluation_results_to_pandas(eval_res_list: List[dict]):
    dataframes = []
    print(eval_res_list)
    for eval_res in eval_res_list:
        df = trackeval_to_pandas(eval_res['evaluation_results'][0]['MotChallenge2DBox']['default_tracker'])
        df['TrackerName'] = eval_res['tracker_name']
        df['tracking_results_hash'] = eval_res['tracking_results_hash']
        df['eval_results_hash'] = eval_res['hash']
        df['fps'] = eval_res['fps']
        df['det_fps'] = eval_res['det_fps']
        df['seq_name'] = df.index
        trackeval_data = load_trackeval_evaluation_data([eval_res], "pedestrian")[eval_res['hash']]
        df['HOTA'] = trackeval_data['HOTA']
        df['DetA'] = trackeval_data['DetA']
        df['AssA'] = trackeval_data['AssA']
        df['DetRe'] = trackeval_data['DetRe']
        df['DetPr'] = trackeval_data['DetPr']
        df['AssRe'] = trackeval_data['AssRe']
        df['AssPr'] = trackeval_data['AssPr']
        df['LocA'] = trackeval_data['LocA']

        dataframes.append(df)

    dataframe = pd.concat(dataframes)
    dataframe.set_index('eval_results_hash')

    return dataframe


def get_linspaced_metric_names(metric: str):
    return [metric + '___' + str(i) for i in range(5, 96, 5)]
