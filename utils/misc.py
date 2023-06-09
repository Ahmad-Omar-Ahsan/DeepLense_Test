"""Miscellaneous helper functions."""

import torch
from torch import nn, optim
import numpy as np
import random
import os
import wandb
from models import ConvNext, ConvNext_backbone_model, ViT


def seed_everything(seed: str) -> None:
    """Set manual seed.
    Args:
        seed (int): Supplied seed.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set seed {seed}")


def count_params(model: nn.Module) -> int:
    """Counts number of parameters in a model.
    Args:
        model (torch.nn.Module): Model instance for which number of params is to be counted.
    Returns:
        int: Parameter count.
    """

    return sum(map(lambda p: p.data.numel(), model.parameters()))


def calc_step(epoch: int, n_batches: int, batch_index: int) -> int:
    """Calculates current step.
    Args:
        epoch (int): Current epoch.
        n_batches (int): Number of batches in dataloader.
        batch_index (int): Current batch index.
    Returns:
        int: Current step.
    """
    return (epoch - 1) * n_batches + (1 + batch_index)


def log(log_dict: dict, step: int, config: dict) -> None:
    """Handles logging for metric tracking server, local disk and stdout.
    Args:
        log_dict (dict): Log metric dict.
        step (int): Current step.
        config (dict): Config dict.
    """

    # send logs to wandb tracking server
    if config["exp"]["wandb"]:
        wandb.log(log_dict, step=step)

    log_message = f"Step: {step} | " + " | ".join(
        [f"{k}: {v}" for k, v in log_dict.items()]
    )

    # write logs to disk
    if config["exp"]["log_to_file"]:
        log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

        with open(log_file, "a+") as f:
            f.write(log_message + "\n")

    # show logs in stdout
    if config["exp"]["log_to_stdout"]:
        print(log_message)


def get_model(model_config: dict) -> nn.Module:
    """Creates model from config dict.
    Args:
        model_config (dict): Dict containing model config params. If the "name" key is not None, other params are ignored.
    Returns:
        nn.Module: Model instance.
    """
    if model_config["type"] == "ConvNext":
        return ConvNext(**model_config["ConvNext"])
    elif model_config["type"] == "ConvNext_backbone":
        return ConvNext_backbone_model(**model_config["ConvNext_backbone"])
    elif model_config["type"] == "ViT":
        return ViT(**model_config["ViT"])
    


def save_model(
    epoch: int,
    val_acc: float,
    save_path: str,
    net: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    log_file: str = None,
) -> None:
    """Saves checkpoint.
    Args:
        epoch (int): Current epoch.
        val_acc (float): Validation accuracy.
        save_path (str): Checkpoint path.
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler. Defaults to None.
        log_file (str, optional): Log file. Defaults to None.
    """

    ckpt_dict = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else optimizer,
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else scheduler,
    }

    torch.save(ckpt_dict, save_path)

    log_message = f"Saved {save_path} with accuracy {val_acc}."
    print(log_message)

    if log_file is not None:
        with open(log_file, "a+") as f:
            f.write(log_message + "\n")



def generate_model_outputs(model : nn.Module, test_loader, num_classes: 3):
    model.eval()
    true_labels = []
    class_probs = []
    for i in range(num_classes):
        class_probs.append([])

    with torch.no_grad():
        for i, (sample, labels) in enumerate(test_loader):
            batch_outputs = model(sample)
            batch_probs = torch.softmax(batch_outputs, dim=1)
            true_labels.extend(labels.numpy())
            for i in range(num_classes):
                class_probs[i].extend(batch_probs[:,i].numpy())

    return true_labels, class_probs