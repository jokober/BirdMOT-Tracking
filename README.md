
## Prepare local_data directory


## Environment
Set the SAHI_TRACKING_DATA_PATH pointing to your local_data folder.

```
export SAHI_TRACKING_DATA_PATH=/data/path/to/local_data
```

Alternatively, you can add the environment variable permanently to your .bashrc or .zshrc file, depending on what shell you are using.

## Install dependencies

``
pip install torch, lap
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox pandas xmltodict
pip install git+https://github.com/noahcao/OC_SORT.git
```

```
poetry install
```

## Install additional trackers
```
cd src/sahi_tracking/trackers/
git clone https://github.com/abewley/sort.git
```

# Tracker Configurations
## OC_SORT
        "tracker_type": "oc_sort",
        "tracker_config": {
          "det_thresh": ,
          "max_age": 30,
          "min_hits": 3,
          "iou_threshold": 0.3,
          "delta_t": 3,
          "asso_func": "iou",
          "inertia": 0.2,
          "use_byte": false

"iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}

## ByteTrack
frame_rate
track_thresh
track_buffer
match_thresh # Used for high score detection

## SORT
 max_age=1,
 min_hits=3,
 iou_threshold=0.3

## V-IOU
        "tracker_type": "vio",
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames. Tracks below this length are not initialized.
         ttl (float): maximum number of frames to perform visual tracking.
                      this can fill 'gaps' of up to 2*ttl frames (ttl times forward and backward).
         tracker_type (str): name of the visual tracker to use. see VisTracker for more details. Can be one of following: BOOSTING, MIL, KCF, KCF2, TLD, MEDIANFLOW, GOTURN, MOSSE
         keep_upper_height_ratio (float): float between 0.0 and 1.0 that determines the ratio of height of the object
                                          to track to the total height of the object used for visual tracking.
        }