'Shared imports, constants, default paths and random seeds.'

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# The target dataset: every source in it counts (EH104 tiles are no longer singled out).
DATASET_NAME = "split_80_10_10"
LOCAL_DATASET_PATH = Path(r"E:\STRUCTURAL ENGINEERING\thesis\crackdetection\DATASETS") / DATASET_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
# Encoders of segmentation_models_pytorch (smp) this trainer is meant for. `tu-<name>` is any timm
# model; the others are smp's own. --encoder takes every smp encoder name, not only these.
DEFAULT_ENCODER = "tu-convnext_tiny"
ENCODER_EXAMPLES = (
    "tu-convnext_tiny", "tu-convnext_small", "tu-convnextv2_tiny",
    "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "tu-tf_efficientnetv2_s",
)
METRIC_NAMES = ("precision", "recall", "f1", "iou", "tol_precision", "tol_recall", "tol_f1", "accuracy")
# Pixel counts accumulated per threshold; "near_*" are the 2-px tolerant hits.
COUNT_NAMES = ("tp", "fp", "fn", "near_label", "predicted", "near_prediction", "labelled", "pixels")
BASIC_METRICS = ("precision", "recall", "f1", "iou", "accuracy")  # logged for train and validation at 0.5


def find_dataset_root(base: Path = KAGGLE_INPUT_ROOT) -> Path:
    """Find the dataset among the Kaggle inputs.

    The folder named DATASET_NAME is preferred; otherwise the first input with
    the expected paired split layout is used.
    """
    if base.is_dir():
        candidates = [
            train_directory.parent for train_directory in sorted(base.rglob("train"))
            if (train_directory / "images").is_dir() and (train_directory / "masks").is_dir()
        ]
        named = [candidate for candidate in candidates if candidate.name == DATASET_NAME]
        if candidates:
            return (named or candidates)[0]
    return LOCAL_DATASET_PATH


DEFAULT_DATASET_PATH = find_dataset_root()
DEFAULT_OUTPUT_PATH = (
    KAGGLE_WORKING_ROOT / "unetpp_hairline_output"
    if KAGGLE_WORKING_ROOT.is_dir() else Path("unetpp_hairline_output")
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # NumPy's global RNG is not reseeded per DataLoader worker by PyTorch.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(0)


__all__ = [
    'argparse', 'copy', 'csv', 'json', 'math', 'os', 'random', 're', 'sys', 'time', 'Counter', 'defaultdict',
    'dataclass', 'field', 'Path', 'cv2', 'np', 'torch', 'nn', 'F', 'DataLoader', 'Dataset',
    'WeightedRandomSampler', 'IMAGE_EXTENSIONS', 'DATASET_NAME', 'LOCAL_DATASET_PATH', 'KAGGLE_INPUT_ROOT',
    'KAGGLE_WORKING_ROOT', 'DEFAULT_ENCODER', 'ENCODER_EXAMPLES', 'METRIC_NAMES', 'COUNT_NAMES', 'BASIC_METRICS',
    'find_dataset_root', 'DEFAULT_DATASET_PATH', 'DEFAULT_OUTPUT_PATH', 'seed_everything', 'seed_worker',
]
