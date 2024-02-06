import numpy as np
from yupi.core.featurizers import DistanceFeaturizer, DisplacementFeaturizer


def distance_stats(traj_list):
    distances = DistanceFeaturizer(0).featurize(traj_list)
    displacements = DisplacementFeaturizer(0).featurize(traj_list)
    track_length = np.array([len(traj.r) for traj in traj_list])
    track_length = track_length.reshape(-1, 1)
    print("asdasd")
    return {
        "min": np.min(distances),
        "max": np.max(distances),
        "mean": np.mean(distances)

    }
