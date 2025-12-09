import argparse
import logging
import os
from pathlib import Path

import numpy as np
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Logger:
    SUM_FREQ = 100

    def __init__(self, save_path):
        # self.model = model
        # self.scheduler = scheduler
        # self.args = args

        self.save_path = save_path
        self.total_steps = 0
        self.running_loss = {}

        self.writer = SummaryWriter(log_dir=self.save_path)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}] ".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.save_path)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, task):
        self.total_steps += 1

        for key in metrics:
            task_key = str(key) + "_" + task
            if task_key not in self.running_loss:
                self.running_loss[task_key] = 0.0

            self.running_loss[task_key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.save_path)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


class TrainLogger:
    SUM_FREQ = 100

    def __init__(self, save_path):
        # self.model = model
        # self.scheduler = scheduler
        self.save_path = save_path
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=self.save_path)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / TrainLogger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}] ".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.save_path)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / TrainLogger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, task):
        self.total_steps += 1

        for key in metrics:
            task_key = str(key) + "_" + task
            if task_key not in self.running_loss:
                self.running_loss[task_key] = 0.0

            self.running_loss[task_key] += metrics[key]

        if self.total_steps % TrainLogger.SUM_FREQ == TrainLogger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.save_path)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
