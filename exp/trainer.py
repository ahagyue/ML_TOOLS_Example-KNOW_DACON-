from collections import defaultdict
from tqdm import tqdm
import time

import torch


class MetricMonitor:
    def __init__(self, float_precision=3, early_stop_step=0):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

class Trainer:

    def __init__(
            self, model, criterion, optimizer,
            train_dataloader, val_dataloader,
            train_target_format, train_output_format, val_target_format, val_output_format, params
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader, self.val_dataloader = train_dataloader, val_dataloader
        self.train_target_format, self.train_output_format = train_target_format, train_output_format
        self.val_target_format, self.val_output_format = val_target_format, val_output_format
        self.params = params

    def train_epoch(self, epoch):
        start = time.time()
        metric_monitor = MetricMonitor()
        self.model.train()
        stream = tqdm(self.train_dataloader)
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(self.params["device"], non_blocking=True)
            target = self.train_target_format(target.to(self.params["device"], non_blocking=True))
            output = self.train_output_format(self.model(images))
            loss = self.criterion(output, target)
            metric_monitor.update("Loss", loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            stream.set_description(
                "Epoch: {epoch}. Train.      {metric_monitor} , time : {time}".format(epoch=epoch,
                                                                                      metric_monitor=metric_monitor,
                                                                                      time=time.time() - start)
            )
        return metric_monitor


    def validation_epoch(self, epoch):
        start = time.time()
        metric_monitor = MetricMonitor()
        self.model.eval()
        stream = tqdm(self.val_dataloader)
        with torch.no_grad():
            for i, (images, target) in enumerate(stream, start=1):
                images = images.to(self.params["device"], non_blocking=True)
                target = self.val_target_format(target.to(self.params["device"], non_blocking=True))
                output = self.val_output_format(self.model(images))
                loss = self.criterion(output, target)
                metric_monitor.update("Loss", loss.item())
                stream.set_description(
                    "Epoch: {epoch}. Validation. {metric_monitor} , time : {time}".format(epoch=epoch,
                                                                                          metric_monitor=metric_monitor,
                                                                                          time=time.time() - start)
                )
        return metric_monitor
