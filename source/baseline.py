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

    clean_model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)),
                           optimizer_fn="adam",
                           model_name="clean_model")
    clean_network = fe.Network(ops=[
        Watch(inputs="x"),
        ModelOp(model=clean_model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        FGSM(data="x", loss="base_ce", outputs="x_adverse", epsilon=epsilon, mode="!train"),
        ModelOp(model=clean_model, inputs="x_adverse", outputs="y_pred_adv", mode="!train"),
        UpdateOp(model=clean_model, loss_name="base_ce")
    ])
    clean_traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="clean_accuracy"),
        Accuracy(true_key="y", pred_key="y_pred_adv", output_name="adversarial_accuracy"),
        BestModelSaver(model=clean_model, save_dir=save_dir, metric="base_ce", save_best_mode="min"),
    ]
    clean_estimator = fe.Estimator(pipeline=pipeline,
                                   network=clean_network,
                                   epochs=epochs,
                                   traces=clean_traces,
                                   log_steps=1000)
    return clean_estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
