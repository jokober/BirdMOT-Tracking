import argparse
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from sahi_tracking.experiments_framework.run_experiment_framework import run_experiment_framework
from sahi_tracking.helper.config import get_local_data_path
from sahi_tracking_tuning.backsub_tuning.param_spaces import mog2_experiment_config


def backsub_tune_runner(config):
    # experiment_dict = {
    #    "tracker_experiments": [
    #        config['experiment_config']
    #    ]
    # }
    eval_results = run_experiment_framework(config['tracking_dataset_dict'], config['detection_params_dict'],
                                            config["tracking_params_dict"],
                                            model_path=config["model_path"],
                                            device=config["device"],
                                            overwrite_existing=config["overwrite_existing"],
                                            read_only=True)

    if False:
        if tuning_experiments_path.exists():
            with open(tuning_experiments_path) as f:
                tune_experiments = json.load(f)
        else:
            tune_experiments = {
                "tracker_experiments": [
                ]
            }
        tune_experiments['tracker_experiments'].append(config['experiment_config'])
        with open(tuning_experiments_path, 'w') as f:
            json.dump(tune_experiments, f)

    sequences_no_overlap = [
        # "C0054_823576_823631", #

        "C0054_811979_812013",
        "C0054_815863_815880",
        "C0054_823576_823631",
        "C0085_125820_125828",
        "C0085_204724_204728",
        "C0085_210357_210373",
        "C0085_234269_234281",
        "C0085_251517_251527",
        "C0085_260828_260834",
        "C0085_267724_267743",
        "C0085_274156_274172",
        "C0085_276602_276613",
        "C0085_280894_280899",
        "MOT-1654349058_zoex_1600166_1600619",
        "MOT-1654349058_zoex_2199821_2200069"
    ]

    sequences_varying_object_size = [
        "C0054_721231_721301",
        "C0085_210357_210373"
    ]

    sequences_small_objects = [
        "MOT-1654349058_zoex_2199821_2200069",
        "MOT-1654349058_zoex_2583995_2584133",
        "MOT-1654911941_zoex_0446964_0447233",
        "MOT-1656471341_zoex_0149757_0149861",
        "MOT-1656677441_zoex_0453755_0453959",
    ]

    sequences_non_linear_motion = [
    ]

    challenging_seqs_set = set(
        sequences_no_overlap + sequences_varying_object_size + sequences_small_objects + sequences_non_linear_motion)

    HOTA_challenging_seqs = np.average([eval_results["evaluation_results"][0]['MotChallenge2DBox']['default_tracker'][
                                            seq]['pedestrian']['HOTA']['HOTA(0)']
                                        for seq in challenging_seqs_set])

    AssA_challenging_seqs = np.average([eval_results["evaluation_results"][0]['MotChallenge2DBox']['default_tracker'][
                                            seq]['pedestrian']['HOTA']['AssA'][0]
                                        for seq in challenging_seqs_set])
    AssRe_challenging_seqs = np.average([eval_results["evaluation_results"][0]['MotChallenge2DBox']['default_tracker'][
                                             seq]['pedestrian']['HOTA']['AssRe'][0]
                                         for seq in challenging_seqs_set])
    AssPr_challenging_seqs = np.average([eval_results["evaluation_results"][0]['MotChallenge2DBox']['default_tracker'][
                                             seq]['pedestrian']['HOTA']['AssPr'][0]
                                         for seq in challenging_seqs_set])

    DetA_challenging_seqs = np.average([eval_results["evaluation_results"][0]['MotChallenge2DBox']['default_tracker'][
                                            seq]['pedestrian']['HOTA']['DetA'][0]
                                        for seq in challenging_seqs_set])

    DetRe_challenging_seqs = np.average([eval_results["evaluation_results"][0]['MotChallenge2DBox']['default_tracker'][
                                             seq]['pedestrian']['HOTA']['DetRe'][0]
                                         for seq in challenging_seqs_set])

    DetPr_challenging_seqs = np.average([eval_results["evaluation_results"][0]['MotChallenge2DBox']['default_tracker'][
                                             seq]['pedestrian']['HOTA']['DetPr'][0]
                                         for seq in challenging_seqs_set])

    LocA_challenging_seqs = np.average([eval_results["evaluation_results"][0]['MotChallenge2DBox']['default_tracker'][
                                            seq]['pedestrian']['HOTA']['LocA'][0]
                                        for seq in challenging_seqs_set])

    train.report(
        {
            "HOTA": eval_results['evaluation_results_summary']['HOTA'],
            "DetA": eval_results['evaluation_results_summary']['DetA'],
            "DetPr": eval_results['evaluation_results_summary']['DetPr'],
            "DetRe": eval_results['evaluation_results_summary']['DetRe'],
            "AssA": eval_results['evaluation_results_summary']['AssA'],
            "AssPr": eval_results['evaluation_results_summary']['AssPr'],
            "AssRe": eval_results['evaluation_results_summary']['AssRe'],
            "LocA": eval_results['evaluation_results_summary']['LocA'],
            "MOTA": eval_results['evaluation_results_summary']['MOTA'],
            "MOTP": eval_results['evaluation_results_summary']['MOTP'],
            "MODA": eval_results['evaluation_results_summary']['MODA'],
            "CLR_Re": eval_results['evaluation_results_summary']['CLR_Re'],
            "CLR_Pr": eval_results['evaluation_results_summary']['CLR_Pr'],
            "IDF1": eval_results['evaluation_results_summary']['IDF1'],
            "IDR": eval_results['evaluation_results_summary']['IDR'],
            "IDP": eval_results['evaluation_results_summary']['IDP'],
            "HOTA_challenging_seqs": HOTA_challenging_seqs,
            "AssA_challenging_seqs": AssA_challenging_seqs,
            "DetA_challenging_seqs": DetA_challenging_seqs,
            "AssRe_challenging_seqs": AssRe_challenging_seqs,
            "AssPr_challenging_seqs": AssPr_challenging_seqs,
            "DetRe_challenging_seqs": DetRe_challenging_seqs,
            "DetPr_challenging_seqs": DetPr_challenging_seqs,
            "LocA_challenging_seqs": LocA_challenging_seqs,
            "done": True,
        }
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--tracking_config_path", type=Path, required=True)
    parser.add_argument("--tracking_dataset_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--overwrite_existing", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", type=str, required=False, default='cpu')
    parser.add_argument("--samples", type=int, required=False, default=300)
    parser.add_argument("--nproc", type=int, required=False, default=8)
    parser.add_argument("--metric", type=str, required=False, default='HOTA')
    args = parser.parse_args()

    ray.init(num_cpus=args.nproc)

    tracking_dataset_path = args.tracking_dataset_path.resolve()
    assert tracking_dataset_path, f"Tracking dataset {tracking_dataset_path} does not exist"
    with open(tracking_dataset_path) as json_file:
        tracking_dataset_dict = json.load(json_file)

    tracking_config_path = args.tracking_config_path.resolve()
    assert tracking_config_path.exists(), f"Sahi predictions params {tracking_config_path} does not exist"
    with open(tracking_config_path) as json_file:
        tracking_params_dict = json.load(json_file)

    config = {
        "detection_params_dict": None,
        "tracking_params_dict": tracking_params_dict,
        "tracking_dataset_dict": tracking_dataset_dict,
        "model_path": args.model_path,
        "device": args.device,
        "overwrite_existing": args.overwrite_existing
    }

    # Prepare Param Space
    if args.algo == 'MOG2':
        config['detection_params_dict'] = mog2_experiment_config
    else:
        raise NotImplementedError("The background substraction algo is not implemented.")

    tuner = tune.Tuner(
        backsub_tune_runner,
        tune_config=tune.TuneConfig(
            metric=args.metric,
            mode="max",
            scheduler=ASHAScheduler(),
            num_samples=args.samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    print(results.get_best_result())
    print(results.get_dataframe())

    result_df = results.get_dataframe()
    result_df = result_df.sort_values(by=[args.metric], ascending=False)[:800]

    tune_result_path = get_local_data_path() / "work_dir/tuning"
    tune_result_path.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(tune_result_path / f"tune_{args.metric}_{args.algo}.csv", index=False)
