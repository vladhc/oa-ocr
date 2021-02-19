import os

import pathlib
from glob import glob
from typing import List

import tensorflow as tf
from tqdm.auto import tqdm


CHECKPOINT_NAME = "model.{epoch:03d}.hdf5"
LOGS_DIR = "logs"
TRAIN_DIR = "train"
DATASET_DIR = pathlib.Path('dataset/synth-text')


def epoch_from_checkpoint(checkpoint: str) -> int:
    name = os.path.basename(checkpoint)
    return int(name.split(".")[1])


def get_checkpoints(model_id: str) -> List[str]:
    checkpoints = glob(os.path.join(TRAIN_DIR, model_id, "model.*.hdf5"))
    return sorted(list(checkpoints))


def get_latest_checkpoint(model_id: str) -> str:
    assert model_id
    checkpoints = get_checkpoints(model_id)
    latest_epoch = 0
    latest_checkpoint = ""

    for checkpoint in checkpoints:
        epoch = epoch_from_checkpoint(checkpoint)
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_checkpoint = checkpoint

    return latest_checkpoint


class TQDMProgressBar(tf.keras.callbacks.Callback):
    """Progress bar for model training and evaluation."""

    def __init__(self, loss_agg, accuracy=None):
        super().__init__()
        self.max_size = 0
        self.update_max_size = True
        self.loss_agg = loss_agg
        self.accuracy = accuracy

    def on_epoch_begin(self, epoch, logs=None):
        self.tqdm.reset(total=self.max_size)
        self.tqdm.set_description("Epoch {}".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        self.update_max_size = False

    def on_train_batch_begin(self, step, logs=None):
        size = logs["size"]
        if self.update_max_size:
            self.max_size += size
        self.tqdm.update(size)

    def on_test_batch_begin(self, step, logs=None):
        self.on_train_batch_begin(step, logs)

    def on_train_batch_end(self, step, logs=None):
        self.tqdm.set_postfix(
            loss="{:.4f}".format(self.loss_agg.result().numpy()),
            refresh=False)

    def on_test_batch_end(self, step, logs=None):
        self.tqdm.set_postfix(
            accuracy="{:.3f}".format(self.accuracy.result().numpy()),
            refresh=False)

    def on_train_begin(self, logs=None):
        self.tqdm = tqdm(
            unit="records",
            smoothing=1.,
            dynamic_ncols=True,
            total=self.max_size)

    def on_test_begin(self, logs=None):
        self.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.tqdm.close()

    def on_test_end(self, logs=None):
        self.on_train_end(logs)
