from typing import Tuple, Mapping, List
import pathlib
import functools
import random

import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE


def decode_fn(record_bytes) -> Mapping[str, tf.Tensor]:
    return tf.io.parse_single_example(record_bytes, {
        'img': tf.io.FixedLenFeature([], dtype=tf.string),
        'prob_map': tf.io.FixedLenFeature([], dtype=tf.string),
        'threshold_map': tf.io.FixedLenFeature([], dtype=tf.string),
    })

def decode_image(record: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    return {
        'img': tf.io.decode_jpeg(record['img']),
        'prob_map': tf.io.decode_png(record['prob_map']),
        'threshold_map': tf.io.decode_png(record['threshold_map']),
    }

def resize(
        record: Mapping[str, tf.Tensor],
        target_size: Tuple[int, int]) -> Mapping[str, tf.Tensor]:

    resize_fn = functools.partial(
        tf.image.resize,
        size=target_size)
    return {
        'img': resize_fn(record['img']),
        'prob_map': resize_fn(record['prob_map']),
        'threshold_map': resize_fn(record['threshold_map']),
    }

def cast(record: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    return {
        'img': tf.cast(record['img'], tf.float32) / 255.,
        'prob_map': tf.cast(record['prob_map'], tf.float32) / 255.,
        'threshold_map': tf.cast(record['threshold_map'], tf.float32) / 255.,
    }

def create_dataset(
        fpaths: List[pathlib.Path],
        img_size: Tuple[int, int],
        train: bool,
        batch_size=32) -> tf.data.Dataset:

    tfrecords = list([str(p) for p in fpaths])
    assert tfrecords, \
        "No tfrecord files satisfy filemasks {}".format(filemasks)

    random.shuffle(tfrecords)
    tfrecord_files = tf.data.Dataset.from_tensor_slices(tfrecords)
    if train:
        tfrecord_files = tfrecord_files.shuffle(
            len(tfrecords),
            reshuffle_each_iteration=True)

    dataset = tfrecord_files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=False)

    if train:
        dataset = dataset.shuffle(
            batch_size * 30,
            reshuffle_each_iteration=True)

    dataset = dataset.map(
        decode_fn,
        num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(
        decode_image,
        num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(
        functools.partial(resize, target_size=img_size),
        num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(
        cast,
        num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset
