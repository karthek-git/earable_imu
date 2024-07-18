import os
import pathlib
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class OverFitMonCB(tf.keras.callbacks.Callback):
    """Monitor Overfitting."""

    def on_epoch_end(self, epoch, logs):
        """Print the loss ratio."""
        print(f"\tval_loss/loss: {logs['val_loss']/logs['loss']}")


TBOARD_ROOT_LOGDIR = "artifacts/tboard/"


def get_tboard_logdir():
    """Get unique logdir name for each run."""
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")

    return os.path.join(TBOARD_ROOT_LOGDIR, run_id)


def split_ds(ds_lbls, cols):
    n_cols = len(cols)
    x_train, x_val, x_test, y_train, y_val, y_test = np.empty(
        (0, n_cols)), np.empty((0, n_cols)), np.empty((0, n_cols)), np.empty(
            (0, )), np.empty((0, )), np.empty((0, ))

    for ds_lbl in ds_lbls:
        x = ds_lbl[cols].values
        y = ds_lbl["Direction"].factorize()[0]
        x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
            x, y, test_size=0.3)
        x_val_s, x_test_s, y_val_s, y_test_s = train_test_split(x_test_s,
                                                                y_test_s,
                                                                test_size=0.33)
        x_train = np.concatenate((x_train, x_train_s))
        x_val = np.concatenate((x_val, x_val_s))
        x_test = np.concatenate((x_test, x_test_s))
        y_train = np.concatenate((y_train, y_train_s))
        y_val = np.concatenate((y_val, y_val_s))
        y_test = np.concatenate((y_test, y_test_s))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test


def reshape_ds(x, y):
    x = np.array(x)
    # n_rows=(x.shape[0]//N_SR)*N_SR
    # x=x[:n_rows].reshape((-1,N_SR*x.shape[1]))
    # y=y[:n_rows][::N_SR].reshape((-1,1))

    return x, y


def tflite_convert(model: tf.keras.Model):
    """Convert model to quantized tflite model with optimizations."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    pathlib.Path("artifacts/models/gic_uint8_v1.tflite").write_bytes(
        tflite_model)

    return tflite_model


def main():
    ds_lbls = []
    dir_name = "ds/ds_labels/"

    for fname in sorted(os.listdir(dir_name)):
        ds_lbls.append(pd.read_csv(dir_name + fname))

    x_cols = ds_lbls[0].columns[8:14]
    print(x_cols)
    x_train, x_val, x_test, y_train, y_val, y_test = split_ds(ds_lbls, x_cols)
    print("No. Train samples:", x_train.shape[0])
    print("No. Val samples:", x_val.shape[0])
    print("No. Test samples:", x_test.shape[0])

    x_train_r, y_train_r = reshape_ds(x_train, y_train)
    x_val_r, y_val_r = reshape_ds(x_val, y_val)
    x_test_r, y_test_r = reshape_ds(x_test, y_test)

    model = tf.keras.models.Sequential(layers=(
        tf.keras.layers.Dense(256,
                              activation=tf.keras.activations.relu,
                              input_shape=(x_train_r.shape[1], )),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(16, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)))

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["acc"])

    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="artifacts/models/model.keras", save_best_only=True)
    cb_earlystop = tf.keras.callbacks.EarlyStopping(patience=16,
                                                    restore_best_weights=True)
    cb_tboard = tf.keras.callbacks.TensorBoard(log_dir=get_tboard_logdir())
    callbacks = (cb_checkpoint, cb_earlystop, cb_tboard, OverFitMonCB())

    model.fit(x_train_r,
              y_train_r,
              epochs=1024,
              callbacks=callbacks,
              validation_data=(x_val_r, y_val_r))
    model.evaluate(x_test_r, y_test_r)
    tflite_convert(model)


if __name__ == "__main__":
    main()
