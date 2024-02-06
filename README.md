BirdMOT Tracking Experiments
========
This repository contains the code used to run experiments on bird tracking for my master thesis. 

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

*Note: This code is not intended to be maintained.*



## Install
### Install using Poetry
```bash
poetry install
```

### Install dependencies

```bash
pip install torch, lap
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox pandas xmltodict
pip install git+https://github.com/noahcao/OC_SORT.git
```

### Install additional trackers
```
cd src/sahi_tracking/trackers/
git clone https://github.com/abewley/sort.git
```
### Environment
Set the SAHI_TRACKING_DATA_PATH pointing to your local_data folder.

```
export SAHI_TRACKING_DATA_PATH=/path/to/local_data
```

Alternatively, you can add the environment variable permanently to your .bashrc or .zshrc file, depending on what shell you are using.
