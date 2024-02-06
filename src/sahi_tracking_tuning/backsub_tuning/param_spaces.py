from ray import tune

mog2_experiment_config = \
    {
        "detector_type": "background_substraction",
        "algo": "MOG2",
        "bbox_thresh": tune.randint(30, 155),
        "nms_thresh": tune.uniform(0.001, 0.005),
        "kernel": tune.randint(2, 20),
        "detectShadows": False,
        "shadow_thresh": tune.uniform(0.2, 0.7),
        "history": tune.randint(2, 10),
        "var_thresh": tune.randint(1, 100),
        "dist2Threshold": 1000
    }

oc_sort_experiment_config = {
    "tracker_type": "oc_sort",
    "tracker_config": {
        "det_thresh": 0.56,
        "max_age": 1,
        "min_hits": 1,
        "iou_threshold": tune.uniform(-0.2, 0.000000001),
        "delta_t": 6,
        "asso_func": "diou",  # tune.choice(['iou','ct_dist', 'ciou', 'giou', 'diou']),
        "inertia": tune.uniform(0.2, 0.7),
        "use_byte": False
    },
    "filter": {
        "filter_type": "simple",
        "filter_config": {
            "min_distance": tune.randint(50, 468),
            "min_displacement": tune.randint(50, 468),
            "min_track_length": tune.randint(2, 10)
        }
    }
}
