import os
import pathlib
from typing import List

import tensorflow as tf
import tensorflow.keras.activations as activations

from ocr import ImageEncoder
from ocr import FeatureToImage
from ocr import create_dataset

import commons


assert tf.config.list_physical_devices('GPU')


# IMG_SIZE = (480, 640)
IMG_SIZE = (224, 320)  # (480, 640)  # should be dividable by 32
BATCH_SIZE = 32


def train(epochs: int, batch_size: int):
    model_id = "db"
    strategy = tf.distribute.MirroredStrategy()

    batch_size = batch_size * strategy.num_replicas_in_sync
    dataset = create_dataset(
        list(commons.DATASET_DIR.glob('train-*.tfrecord')),
        img_size=IMG_SIZE,
        batch_size=batch_size)
    # If not DATA, then we are getting empty batch on one of the workers
    dataset.options().experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA
    dataset = strategy.experimental_distribute_dataset(
        dataset)

    with strategy.scope():
        img_input = tf.keras.Input(
            shape=IMG_SIZE + (3,),
            dtype=tf.float32,
            name='image_input')
        prob_map_gt = tf.keras.Input(
            shape=IMG_SIZE,
            dtype=tf.float32,
            name='prob_map_gt')
        threshold_map_gt = tf.keras.Input(
            shape=IMG_SIZE,
            dtype=tf.float32,
            name='threshold_map_gt')

        encoder = ImageEncoder()
        to_prob_map = FeatureToImage()
        to_threshold_map = FeatureToImage()

        x = encoder(img_input)
        prob_map = to_prob_map(x)
        threshold_map = to_threshold_map(x)
        k = 50
        binary_map = activations.sigmoid(k * (prob_map - threshold_map))

        loss_prob_map = tf.keras.losses.BinaryCrossentropy(
            name='loss_prob_map',
            reduction=tf.keras.losses.Reduction.NONE,
        )(prob_map_gt, prob_map)
        binary_map_gt = activations.sigmoid(
            k * (prob_map_gt - threshold_map_gt))

        loss_binary_map = tf.keras.losses.BinaryCrossentropy(
            name='loss_binary_map',
            reduction=tf.keras.losses.Reduction.NONE,
        )(binary_map_gt, binary_map)

        loss_threshold_map = tf.keras.losses.MeanAbsoluteError(
            name='loss_threshold_map',
            reduction=tf.keras.losses.Reduction.NONE,
        )(threshold_map_gt, threshold_map)
        loss = loss_prob_map + 1. * loss_binary_map + 10. * loss_threshold_map
        model = tf.keras.Model(
            inputs=[img_input, prob_map_gt, threshold_map_gt],
            outputs=[loss],
        )

        optimizer = tf.keras.optimizers.Adam()

    epoch = 0
    checkpoint = commons.get_latest_checkpoint(model_id)
    if checkpoint:
        epoch = commons.epoch_from_checkpoint(checkpoint)
        model.load_weights(checkpoint)
        print("Continue training from the {} epoch".format(epoch))

    @tf.function
    def train_step(batch):
        img_shape = batch['img'].shape.as_list()
        prob_map_shape = batch['prob_map'].shape.as_list()
        threshold_map_shape = batch['threshold_map'].shape.as_list()
        assert img_shape[:3] == prob_map_shape[:3]  # all except channels
        assert img_shape[:3] == threshold_map_shape[:3]  # all except channels
        assert prob_map_shape == threshold_map_shape
        assert img_shape[0] > 0, "Empty batch {}".format(img_shape)
        with tf.GradientTape() as tape:
            loss = model([
                batch['img'],
                batch['prob_map'],
                batch['threshold_map'],
            ], training=True)
            loss = tf.nn.compute_average_loss(
                loss, global_batch_size=batch_size)
            assert loss.shape.as_list() == [], loss.shape
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def distributed_train_step(batch):
        per_replica_loss = strategy.run(train_step, args=(batch,))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_loss,
            axis=None)

    loss_agg = tf.keras.metrics.Mean(name="train_loss")
    train_dir = os.path.join(commons.TRAIN_DIR, model_id)
    logs_dir = os.path.join(commons.LOGS_DIR, model_id)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            save_weights_only=True,
            filepath=os.path.join(train_dir, commons.CHECKPOINT_NAME)),
        tf.keras.callbacks.TensorBoard(
            logs_dir,
            update_freq=100000,
            write_graph=False,
            profile_batch=0),
        commons.TQDMProgressBar(loss_agg),
    ]

    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin()

    with strategy.scope():
        for cur_epoch in range(epoch, epochs):
            for callback in callbacks:
                callback.on_epoch_begin(cur_epoch)
            loss_agg.reset_states()

            for step, batch in enumerate(dataset):
                for callback in callbacks:
                    callback.on_train_batch_begin(step, {"size": batch_size})
                loss = distributed_train_step(batch)
                loss_agg.update_state(loss)
                for callback in callbacks:
                    callback.on_train_batch_end(step, {
                        "loss": loss.numpy(),
                    })

            for callback in callbacks:
                callback.on_epoch_end(
                    cur_epoch,
                    logs={
                        "loss": loss_agg.result(),
                    })


if __name__ == "__main__":
    train(
        epochs=20,
        batch_size=BATCH_SIZE)
