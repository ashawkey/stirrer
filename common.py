import os
import gc
import sys
import cv2
import glob
import time
import signal
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import tensorboardX
from tqdm import tqdm

from collections import OrderedDict
from sklearn import model_selection

fold = 0
workspace = "./workspace/test"

seed = 42
test_size = 0.2
num_splits = 5

PREPROCESS = False
TRAIN = True
PREDICT = True
ENSEMBLE = False
