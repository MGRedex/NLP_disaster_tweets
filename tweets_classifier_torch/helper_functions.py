from torch.utils.tensorboard import SummaryWriter
from torcheval import metrics
from torch.optim import Optimizer
from torch.nn import Module
from .model import *
from .datasets import *
import os
from typing import Dict

def create_run_logger(
        folder: str,
        model: Module
    ) -> SummaryWriter:
    """Creates SummaryWriter for model training run
    that writes into folder\\model_name\\run_number
    where:

        folder - log folder

        model_name - name of model's class

        run_number - next run number (if there are previous run folders) or 0
    """
    run_folder = f"{folder}/{model.__class__.__name__}"
    try:
        run_n = int(max(os.listdir(run_folder)))
        run_n += 1
    except:
        run_n = 0
    return SummaryWriter(f"{run_folder}/{run_n}")
    
def train_step(
        model: TweetsDisasterClassifier,
        data: TweetsV2,
        optimizer: Optimizer,
        loss_fn: Module,
        metrics_dict: Dict[str, metrics.Metric],
        device: str
    ) -> int:
    """Train step for TweetsDisasterClassifier.
    
    Returns:
        Loss
    """
    model.to(device)
    ov_loss = 0
    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)

        y_pred = model(X).squeeze()

        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        ov_loss += loss

        for metric in metrics_dict.values():
            metric.to(device)
            metric.update(y_pred, y)

    return ov_loss / len(data)