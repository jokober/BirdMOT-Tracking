from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from yupi import Trajectory
from yupi.core import featurizers
from yupi.graphics import plot_2d, plot_hist
from tensorflow import keras
import tensorflow as tf

from sahi_tracking.trajectories.from_mot import yupi_traj_from_mot
from sahi_tracking.trajectories.traj_analysis import distance_stats

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

from pactus import Dataset, featurizers
from pactus.models import (
    DecisionTreeModel,
    KNeighborsModel,
    LSTMModel,
    RandomForestModel,
    TransformerModel,
    XGBoostModel,
)


def pactus_train(trajs):
    labels = ["bird" for i in range(len(trajs))]
    ds = Dataset("dummy", trajs, labels)

    traj_count = 200
    first_trajs = ds.trajs[:traj_count]
    plot_2d(first_trajs, legend=False, color="#2288dd", show=False)
    plt.title(f"First {traj_count} trajectories")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.show()

    lengths = np.array([len(traj) for traj in ds.trajs])
    plot_hist(lengths, bins=40, show=False)
    plt.title("Trajectory lengths historgram")
    plt.xlabel("Length")
    plt.show()

    # Classes that are going to be used
    ds = Dataset("dummy", trajs, labels)
    SEED = 0
    # Preprocess the dataset and split it into train and test sets
    train, test = (
        # Remove short and poorly time sampled trajectories
        ds.filter(lambda traj, _: len(traj) > 3)
        # Split the dataset into train and test
        .split(train_size=0.7, random_state=SEED)
    )

    featurizer = featurizers.VelocityFeaturizer()

    model = RandomForestModel(
        featurizer=featurizer,
        max_features=16,
        n_estimators=200,
        bootstrap=False,
        random_state=SEED,
        warm_start=True,
        n_jobs=6,
    )

    print(train.trajs)
    # Train the model
    model.train(data=train, cross_validation=2)

    # Evaluate the model on a test dataset
    evaluation = model.evaluate(test)

    # Show the evaluation results
    evaluation.show()

    featurizer = featurizers.VelocityFeaturizer()
    vectorized_models = [
        RandomForestModel(
            featurizer=featurizer,
            max_features=16,
            n_estimators=200,
            bootstrap=False,
            warm_start=True,
            n_jobs=6,
            random_state=SEED,
        ),
        KNeighborsModel(
            featurizer=featurizer,
            n_neighbors=7,
        ),
        DecisionTreeModel(
            featurizer=featurizer,
            max_depth=7,
            random_state=SEED,
        ),
        # SVMModel(
        #    featurizer=featurizer,
        #    random_state=SEED,
        # ),
        XGBoostModel(
            featurizer=featurizer,
            random_state=SEED,
        ),
    ]

    lstm = LSTMModel(
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
        random_state=SEED,
    )

    model = TransformerModel(
        head_size=512,
        num_heads=4,
        num_transformer_blocks=4,
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        random_state=SEED,
    )

    for model in vectorized_models:
        print(f"\nModel: {model.name}\n")
        model.train(train, cross_validation=5)
        evaluation = model.evaluate(test)
        evaluation.show()

    checkpoint = keras.callbacks.ModelCheckpoint(
        f"partially_trained_model_lstm_{ds.name}.h5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    # lstm.train(train, ds, epochs=20, checkpoint=checkpoint)
    # evaluation = lstm.evaluate(test)
    # evaluation.show()

    checkpoint = keras.callbacks.ModelCheckpoint(
        f"partially_trained_model_transformer_{ds.name}.h5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    # transformer.train(train, ds, epochs=150, checkpoint=checkpoint)
    # evaluation = transformer.evaluate(test)
    # evaluation.show()


if __name__ == "__main__":
    trajs = []
    for path in Path(
            "/media/data/BirdMOT/local_data_tracking/work_dir/datasets/3c3bd4f3e00cac52c398dbf7d4cbced84feb3ea5b0d1b6c0a266fab4b8492e21/MOT/BirdMOT-all/").glob(
        '**/*gt.txt'):
        trajs.extend(yupi_traj_from_mot(path)[1])
        print(path.as_posix())
    distance_stats(trajs)
    pactus_train(trajs)
