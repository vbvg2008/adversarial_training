import os
import tempfile

import fastestimator as fe
import numpy as np
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.backend import argmax, to_tensor
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize, Onehot
from fastestimator.op.tensorop import Average
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.util import ImgData, to_number


def get_estimator(epochs=10, batch_size=50, epsilon=0.04, save_dir=tempfile.mkdtemp()):
    train_data, eval_data = cifar10.load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))])
    model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam", model_name="adv_model")
    network = fe.Network(ops=[
        Watch(inputs="x"),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        FGSM(data="x", loss="base_ce", outputs="x_adverse", epsilon=epsilon),
        ModelOp(model=model, inputs="x_adverse", outputs="y_pred_adv"),
        CrossEntropy(inputs=("y_pred_adv", "y"), outputs="adv_ce"),
        Average(inputs=("base_ce", "adv_ce"), outputs="avg_ce"),
        UpdateOp(model=model, loss_name="avg_ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="clean_accuracy"),
        Accuracy(true_key="y", pred_key="y_pred_adv", output_name="adversarial_accuracy"),
        BestModelSaver(model=model, save_dir=save_dir, metric="base_ce", save_best_mode="min"),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             monitor_names=["base_ce", "adv_ce"],
                             log_steps=1000)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
