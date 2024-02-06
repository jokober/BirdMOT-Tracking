from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from yupi import Trajectory


def yupi_traj_from_mot(mot: Union[Path, ndarray, DataFrame]) -> Tuple[ndarray, List]:
    if isinstance(mot, Path):
        df = pd.read_csv(mot,
                         names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
    elif isinstance(mot, ndarray):
        df = pd.DataFrame(mot,
                          columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
    else:
        raise NotImplementedError

    df['center_point_y'] = df.apply(lambda row: round(row.bb_top + row.bb_height / 2), axis=1)
    df['center_point_x'] = df.apply(lambda row: round(row.bb_left + row.bb_width / 2), axis=1)

    yupi_traj_list = []
    instance_ids = []
    for id in df["id"].unique():
        track_df = df.loc[df['id'] == id]
        dt = track_df["frame"].min()
        # track_df["frame"] = track_df.apply(lambda row: round(row.frame-dt), axis=1)
        track_df = track_df.sort_values(by=['frame'])
        x = track_df["center_point_x"].values.tolist()
        if len(x) >= 2:
            y = track_df["center_point_y"].values.tolist()
            t = track_df["frame"]
            assert (len(x) == len(y))
            assert (len(x) >= 2)
            yupi_traj_list.append(Trajectory(x=x, y=y, t=t, t_0=dt, traj_id=f"{id}"))
            instance_ids.append(id)

    return np.array(instance_ids), yupi_traj_list
