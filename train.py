import time
import csv
import datetime
import torch
import torch.optim
import torch.utils.data

import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from path import Path
from tensorboardX import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

