from typing import Tuple, Mapping
import pathlib
import functools

import tensorflow as tf


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

def resize(record: Mapping[str, tf.Tensor], target_size: Tuple[int, int]) -> Mapping[str, tf.Tensor]:
    resize = functools.partial(
        tf.image.resize,
        size=target_size)
    return {
        'img': resize(record['img']),
        'prob_map': resize(record['prob_map']),
        'threshold_map': resize(record['threshold_map']),
    }

def cast(record: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    return {
        'img': tf.cast(record['img'], tf.float32) / 255.,
        'prob_map': tf.cast(record['prob_map'], tf.float32) / 255.,
        'threshold_map': tf.cast(record['threshold_map'], tf.float32) / 255.,
    }

def create_dataset(
        fpaths: pathlib.Path,
        img_size: Tuple[int, int],
        batch_size=32) -> tf.data.Dataset:

    fpaths = [str(p) for p in fpaths]
    dataset = tf.data.TFRecordDataset(list(fpaths))
    dataset = dataset.map(decode_fn)
    dataset = dataset.map(decode_image)
    dataset = dataset.map(functools.partial(resize, target_size=img_size))
    dataset = dataset.map(cast)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset
