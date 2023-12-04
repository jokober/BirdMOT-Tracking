from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tracking_dataset_path", type=Path, required=True)
    parser.add_argument("--tracking_experiment_path", type=Path, required=True)
    parser.add_argument("--sahi_predictions_params_path", type=Path, required=True)
    parser.add_argument("--sahi_model_path", type=Path, required=True)
    parser.add_argument("--caching",  default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite_existing",  default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", type=str, required=False, default='cpu')
    args = parser.parse_args()

    tracking_dataset_dict, sahi_predictions_params_dict, tracking_experiment_dict = load_config_files(
        args.tracking_dataset_path, args.tracking_experiment_path, args.sahi_predictions_params_path
    )

    run_experiment_framework(tracking_dataset_dict, sahi_predictions_params_dict, tracking_experiment_dict,
        args.sahi_model_path, args.device, args.overwrite_existing)
