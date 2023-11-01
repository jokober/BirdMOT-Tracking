
## Prepare local_data directory


## Environment
Set the SAHI_TRACKING_DATA_PATH pointing to your local_data folder.

```
export SAHI_TRACKING_DATA_PATH=/data/path/to/local_data
```

Alternatively, you can add the environment variable permanently to your .bashrc or .zshrc file, depending on what shell you are using.

## Install dependencies
```
poetry install
```

 pip install git+https://github.com/noahcao/OC_SORT.git

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