import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def local_data_path_fixture(tmp_path_factory):
    fn = tmp_path_factory.mktemp('local_data')
    shutil.copytree((Path(__file__).parents[0] / 'fixtures' / "local_data"), Path(fn), dirs_exist_ok=True)
    return fn


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, local_data_path_fixture):
    monkeypatch.setenv('SAHI_TRACKING_DATA_PATH', local_data_path_fixture.as_posix())
