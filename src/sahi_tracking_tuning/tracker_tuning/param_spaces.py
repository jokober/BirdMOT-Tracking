from ray import tune

sort_experiment_config = {
    "tracker_type": "sort",
    "tracker_config": {
        "max_age": tune.randint(1, 30),
        "min_hits": tune.randint(0, 30),
        "iou_threshold": tune.uniform(0.001, 0.9),
        "alpha": tune.uniform(1, 1.2)
    },
    "filter": {
        "filter_type": "simple",
        "filter_config": {
            "min_distance": tune.randint(50, 468),
            "min_displacement": tune.randint(50, 468),
            "min_track_length": tune.randint(2, 16)
        }
    }
}

oc_sort_experiment_config = {
    "tracker_type": "oc_sort",
    "tracker_config": {
        "det_thresh": tune.uniform(0.28, 0.6),
        "max_age": tune.randint(1, 15),
        "min_hits": tune.randint(0, 30),
        "iou_threshold": tune.uniform(-0.3, 0.000001),
        "delta_t": tune.randint(1, 9),
        "asso_func": "diou",  # tune.choice(['iou','ct_dist', 'ciou', 'giou', 'diou']),
        "inertia": tune.uniform(0.2, 0.7),
        "use_byte": tune.choice([True, False])
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

viou_experiment_config = {
    "tracker_type": "viou",
    "tracker_config": {
        "sigma_l": tune.uniform(0.001, 0.9),
        "sigma_h": tune.uniform(0.001, 0.9),
        "sigma_iou": tune.uniform(0.000000001, 0.9),
        "t_min": tune.randint(0, 30),
        "ttl": tune.randint(1, 30),
        "tracker_type": "KCF",
        "keep_upper_height_ratio": tune.uniform(0.001, 0.9),
    },
    "filter": {
        "filter_type": "simple",
        "filter_config": {
            "min_distance": tune.randint(50, 468),
            "min_displacement": tune.randint(50, 468),
            "min_track_length": tune.randint(2, 16)
        }
    }
}
