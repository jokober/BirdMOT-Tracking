"""'USE_PARALLEL': False,
'NUM_PARALLEL_CORES': 8,
'BREAK_ON_ERROR': True,
'PRINT_RESULTS': True,
'PRINT_ONLY_COMBINED': False,
'PRINT_CONFIG': True,
'TIME_PROGRESS': True,
'OUTPUT_SUMMARY': True,
'OUTPUT_DETAILED': True,
'PLOT_CURVES': True,
Dataset
arguments:
'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
'PRINT_CONFIG': True,  # Whether to print current config
'DO_PREPROC': True,  # Whether to perform preprocessing (never done for 2D_MOT_2015)
'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
Metric
arguments:
'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE']
"""
from typing import List

import trackeval

from sahi_tracking.helper.config import get_evaluation_results_path

default_config = {
    'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
    'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
    'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
    'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
    'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
    'PRINT_CONFIG': True,  # Whether to print current config
    'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
    'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
    'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
    'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
    'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
    'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
    'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
    'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
    # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
    # If True, then the middle 'benchmark-split' folder is skipped for both.
}

def trackeval_evaluate(GT_FOLDER, TRACKERS_FOLDER, OUTPUT_FOLDER, SEQMAP_FILE, BENCHMARK="MOT17", SPLIT_TO_EVAL="all", CLASSES_TO_EVAL=['pedestrian']):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['BENCHMARK'] = BENCHMARK
    dataset_config['SPLIT_TO_EVAL'] = SPLIT_TO_EVAL
    if SEQMAP_FILE:
        dataset_config['SEQMAP_FILE'] = SEQMAP_FILE
    if GT_FOLDER:
        dataset_config['GT_FOLDER'] = GT_FOLDER
    if TRACKERS_FOLDER:
        dataset_config['TRACKERS_FOLDER'] = TRACKERS_FOLDER
    if OUTPUT_FOLDER:
        dataset_config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
    dataset_config['CLASSES_TO_EVAL'] = CLASSES_TO_EVAL


    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.4} #ToDo: Add Threshold as argument?
    #config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs


    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    return evaluator.evaluate(dataset_list, metrics_list)

def load_trackeval_evaluation_data(eval_dict_list: List[dict], cls):
    workdir_eval_path = get_evaluation_results_path()
    data = {}
    for eval_res in eval_dict_list:
        with open(workdir_eval_path / eval_res['hash'] / "default_tracker" / (cls + '_summary.txt')) as f:
            keys = next(f).split(' ')
            done = False
            while not done:
                values = next(f).split(' ')
                if len(values) == len(keys):
                    done = True
            assert 'tracker_name' in eval_res
            data[eval_res['hash']] = dict(zip(keys, map(float, values)))
            print("## " + eval_res['tracker_name'])

    return data