import pickle
from argparse import ArgumentParser
from pathlib import Path

from src.sahi_tracking.experiments_framework.utils import load_config_files
from src.sahi_tracking.helper.config import get_local_data_path


class DataStatePersistance:
    def __init__(self):
        self.local_data_path = get_local_data_path()
        self.state_path: Path = self.local_data_path / 'state.pkl'

        self.state = None
        self.load_state()
    def write_state(self):
        with open(self.state_path, 'wb') as handle:
            pickle.dump(self.state, handle)


    def load_state(self):
        if self.state_path.exists():
            with open(self.state_path, 'rb') as handle:
                self.state = pickle.load(handle)
        else:
            print("state_path does not exist. Creating new state.")
            self.create_new_state()
            self.write_state()


    def create_new_state(self):
        self.state = {
            'datasets': [],
            'tracking_results': [],
            'predictions_results': [],
            'evaluation_results': [],
        }

    def delete_existing(self, key, hash):
        self.load_state()
        print("Deleting existing")

        self.state[key] = [item for item in self.state[key] if item["hash"] != hash]
        assert not hash in [item for item in self.state[key]]
        self.write_state()


    def update_state(self, type, key, value):
        self.load_state()
        if type == 'append':
            self.state[key].append(value)
        else:
            raise NotImplementedError("The type is not implemented.")
        self.write_state()


    def load_data(self, key, hash):
        self.load_state()
        return [item for item in self.state[key] if item["hash"] == hash][0]

    def data_exists(self, key, hash):
        return hash in [item["hash"] for item in self.state[key]]

