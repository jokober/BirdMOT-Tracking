import json
import shutil
from copy import deepcopy
from pathlib import Path

from deepdiff import DeepHash


from sahi_tracking.formats.cocovid import filter_sequences

from sahi_tracking.formats.cocovid2mot import cocovid2mot
from sahi_tracking.helper.config import get_datasets_path, get_coco_files_path

from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance


def find_or_create_dataset(dataset_config: dict, persistence_state: DataStatePersistance, cocovid_img_path: Path = None, overwrite_existing: bool = False):
    dataset_config = deepcopy(dataset_config)
    dataset = {
        'dataset_config': dataset_config,
        'dataset': {},
        'hash': None
    }

    # Create hash of the dataset only based on the dataset config
    deephash_exclude_paths = [
        "root['dataset']",
        "root['hash']",
    ]
    dataset_hash = DeepHash(dataset, exclude_paths=deephash_exclude_paths)[dataset]

    # Delete existing dataset if overwrite_existing is True
    if overwrite_existing:
        persistence_state.state.delete_existing('datasets', dataset_hash)

    # Check if dataset already exists return it if so or create otherwise
    if not persistence_state.data_exists('datasets', dataset_hash):
        dataset_path = get_datasets_path() / dataset_hash
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            print("Dataset already exists. Deleting it and creating a new one.")
        dataset_path.mkdir(exist_ok=True)

        dataset['dataset']['dataset_path'] = dataset_path
        dataset['dataset']['sequences'] = []
        dataset['hash'] = dataset_hash

        for tracking_datasets in dataset_config['tracking_datasets']:
            if 'cocovid_path' in tracking_datasets:
                assert cocovid_img_path is not None, "cocovid_img_path must be set if cocovid_sequence_path is used in dataset config"
                assert tracking_datasets['cocovid_path'], f"CocoVID Sequence {tracking_datasets['cocovid_path']} does not exist"
                with open(get_coco_files_path() / tracking_datasets['cocovid_path']) as json_file:
                    cocovid_dict = json.load(json_file)

                # Filter sequences if filter_sequences is set
                if 'filter_sequences' in tracking_datasets and len(tracking_datasets['filter_sequences']) > 0:
                    cocovid_dict = filter_sequences(cocovid_dict, tracking_datasets['filter_sequences'])

                # Create MOT folder structure and convert coco to mot
                sequences = cocovid2mot(cocovid_dict=cocovid_dict,
                                        out_dir=dataset_path / 'MOT' / f"{dataset['dataset_config']['benchmark_name']}-all",
                                        image_path=cocovid_img_path)
                dataset['dataset']['sequences'].extend(sequences)
            else:
                raise NotImplementedError("Only cocovid_path is supported for now")

            # Create Seqmap file
            seq_map_file = dataset_path / 'seqmap.txt'
            dataset['dataset']['seq_map'] = seq_map_file
            if not seq_map_file.exists():
                seq_map_file.write_text('name\n')
            with open(seq_map_file, 'a') as seqfile:
                for seq in sequences:
                    seqfile.write(f"{seq['name']}\n")

        # Add new dataset to state
        persistence_state.update_state('append', 'datasets', dataset)
    else:
        dataset = persistence_state.load_data('datasets', dataset_hash)

    return dataset